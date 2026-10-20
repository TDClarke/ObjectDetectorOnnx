/*
 * Autopsy Forensic Browser
 *
 * Copyright 2018-2021 Basis Technology Corp.
 * Contact: carrier <at> sleuthkit <dot> org
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package uk.ac.open.ObjectDetectorOnnx;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.openide.util.NbBundle.Messages;
import org.sleuthkit.autopsy.casemodule.Case;
import org.sleuthkit.autopsy.casemodule.NoCurrentCaseException;
import org.sleuthkit.autopsy.coreutils.ImageUtils;
import org.sleuthkit.autopsy.coreutils.Logger;
import org.sleuthkit.autopsy.coreutils.PlatformUtil;
import org.sleuthkit.autopsy.ingest.FileIngestModuleAdapter;
import org.sleuthkit.autopsy.ingest.IngestJobContext;
import org.sleuthkit.autopsy.ingest.IngestMessage;
import org.sleuthkit.autopsy.ingest.IngestModule;
import org.sleuthkit.autopsy.ingest.IngestModuleReferenceCounter;
import org.sleuthkit.autopsy.ingest.IngestServices;
import org.sleuthkit.datamodel.AbstractFile;
import org.sleuthkit.datamodel.Blackboard;
import org.sleuthkit.datamodel.BlackboardArtifact;
import org.sleuthkit.datamodel.BlackboardAttribute;
import org.sleuthkit.datamodel.Score;
import org.sleuthkit.datamodel.TskCoreException;
import org.sleuthkit.datamodel.TskData;

/**
 * File ingest module that detects objects in images using ONNX object
 * detection models (YOLO-style output) run through OpenCV's DNN module.
 *
 * Models are *.onnx files in PlatformUtil.getObjectDetectionClassifierPath().
 * An optional class-name file with the same base name (model.names or
 * model.txt, one label per line) supplies human-readable labels.
 */
public class ObjectDetectorOnnxFileIngestModule extends FileIngestModuleAdapter {

    private final static String MODULE_NAME = ObjectDetectorOnnxModuleFactory.getModuleName();
    private final static Logger logger = Logger.getLogger(ObjectDetectorOnnxFileIngestModule.class.getName());
    private final static int MAX_FILE_SIZE = 100000000;  //Max size of pictures to perform object detection on
    private static final IngestModuleReferenceCounter refCounter = new IngestModuleReferenceCounter();

    /**
     * Input sizes {width, height} tried, in order, when working out what a
     * model expects (640x640 is the YOLOv5/v8/v11 default export). Square sizes
     * come first, then common rectangular exports.
     */
    private static final int[][] CANDIDATE_INPUT_SIZES = {
        {640, 640}, {320, 320}, {416, 416}, {480, 480}, {512, 512}, {576, 576}, {608, 608},
        {704, 704}, {768, 768}, {800, 800}, {960, 960}, {1024, 1024}, {1280, 1280},
        {640, 480}, {480, 640}, {640, 384}, {384, 640}, {640, 352}, {352, 640},
        {320, 240}, {240, 320}, {416, 320}, {320, 416}, {1280, 720}, {720, 1280}
    };
    /** Enough bytes for the longest signature checked (JPEG 2000: 8). */
    private static final int HEADER_BYTES = 12;
    /** Files smaller than this many bytes are not scanned. */
    private static final long MIN_FILE_SIZE = 100;
    /** Also accept anything Autopsy can create a thumbnail for. */
    private static final boolean USE_THUMBNAIL_CHECK = true;
    private static final float CONFIDENCE_THRESHOLD = 0.35f;
    private static final float NMS_THRESHOLD = 0.45f;

    /** A loaded ONNX model with its labels. */
    private static final class OnnxModel {

        final Net net;
        final List<String> labels;
        final int inputWidth;
        final int inputHeight;

        OnnxModel(Net net, List<String> labels, int inputWidth, int inputHeight) {
            this.net = net;
            this.labels = labels;
            this.inputWidth = inputWidth;
            this.inputHeight = inputHeight;
        }
    }

    private long jobId;
    private Map<String, OnnxModel> models;
    private final IngestServices services = IngestServices.getInstance();
    private Blackboard blackboard;

    @Messages({"ObjectDetectorOnnxFileIngestModule.noClassifiersFound.subject=No classifiers found.",
        "# {0} - classifierDir", "ObjectDetectorOnnxFileIngestModule.noClassifiersFound.message=No ONNX models were found in {0}, object detection will not be executed.",
        "ObjectDetectorOnnxFileIngestModule.openCVNotLoaded=OpenCV was not loaded, but is required to run."
    })
    @Override
    public void startUp(IngestJobContext context) throws IngestModule.IngestModuleException {
        jobId = context.getJobId();
        File classifierDir = new File(PlatformUtil.getObjectDetectionClassifierPath());
        models = new LinkedHashMap<>();

        try {
            ObjectDetectorOnnxNativeLoader.load();
        } catch (UnsatisfiedLinkError | RuntimeException ex) {
            String errorMsg = Bundle.ObjectDetectorOnnxFileIngestModule_openCVNotLoaded();
            logger.log(Level.SEVERE, errorMsg, ex);
            throw new IngestModule.IngestModuleException(errorMsg, ex);
        }

        //Load all ONNX models found in PlatformUtil.getObjectDetectionClassifierPath()
        if (classifierDir.exists() && classifierDir.isDirectory()) {
            File[] files = classifierDir.listFiles();
            if (files != null) {
                for (File modelFile : files) {
                    if (modelFile.isFile() && modelFile.getName().toLowerCase().endsWith(".onnx")) {
                        try {
                            Net net = Dnn.readNetFromONNX(modelFile.getAbsolutePath());
                            if (net.empty()) {
                                logger.log(Level.WARNING, "ONNX model is empty: " + modelFile);
                                continue;
                            }
                            int[] inputSize = resolveInputSize(net);
                            if (inputSize == null) {
                                logger.log(Level.WARNING, "Could not determine a supported input size for ONNX model " + modelFile);
                                continue;
                            }
                            logger.log(Level.INFO, "Loaded ONNX model " + modelFile.getName()
                                    + " with input size " + inputSize[0] + "x" + inputSize[1]);
                            models.put(modelFile.getName(), new OnnxModel(net, loadLabels(modelFile), inputSize[0], inputSize[1]));
                        } catch (CvException ex) {
                            logger.log(Level.WARNING, "Unable to load ONNX model " + modelFile, ex);
                        }
                    }
                }
            }
        } else {
            throw new IngestModule.IngestModuleException("Unable to load classifiers for object detection module.");
        }
        if (refCounter.incrementAndGet(jobId) == 1 && models.isEmpty()) {
            services.postMessage(IngestMessage.createWarningMessage(ObjectDetectorOnnxModuleFactory.getModuleName(),
                    Bundle.ObjectDetectorOnnxFileIngestModule_noClassifiersFound_subject(),
                    Bundle.ObjectDetectorOnnxFileIngestModule_noClassifiersFound_message(PlatformUtil.getObjectDetectionClassifierPath())));
        }
        try {
            blackboard = Case.getCurrentCaseThrows().getSleuthkitCase().getBlackboard();
        } catch (NoCurrentCaseException ex) {
            throw new IngestModule.IngestModuleException("Exception while getting open case.", ex);
        }
    }

    /**
     * Works out the input size a model accepts by running a blank image through
     * it at each candidate size. Dynamic-shape models accept the first
     * candidate (640x640); fixed-shape models reject every size but their own
     * with a CvException, so the first size that runs cleanly is the one to use.
     *
     * @return {width, height}, or null if none of the candidates worked
     */
    private static int[] resolveInputSize(Net net) {
        for (int[] candidate : CANDIDATE_INPUT_SIZES) {
            int width = candidate[0];
            int height = candidate[1];
            Mat blank = Mat.zeros(height, width, CvType.CV_8UC3); // rows = height, cols = width
            Mat blob = null;
            List<Mat> outs = new ArrayList<>();
            try {
                blob = Dnn.blobFromImage(blank, 1.0 / 255.0, new Size(blank.cols(), blank.rows()), new Scalar(0, 0, 0), true, false);
                net.setInput(blob);
                net.forward(outs, net.getUnconnectedOutLayersNames());
                if (!outs.isEmpty()) {
                    return new int[]{width, height};
                }
            } catch (CvException ex) {
                // wrong size for this model, try the next candidate
            } finally {
                blank.release();
                if (blob != null) {
                    blob.release();
                }
                for (Mat m : outs) {
                    m.release();
                }
            }
        }
        return null;
    }

    /**
     * Decides whether to attempt detection on the file: it must be at least
     * MIN_FILE_SIZE bytes and either be thumbnail-capable or start with a known
     * image signature.
     *
     * @param file file to be checked
     * @return true if the file should be scanned
     */
    private static boolean isImageFile(AbstractFile file) {
        if (file.getSize() < MIN_FILE_SIZE) {
            return false;
        }

        if (USE_THUMBNAIL_CHECK && ImageUtils.isImageThumbnailSupported(file)) {
            return true;
        }

        byte[] header = new byte[HEADER_BYTES];
        int bytesRead;
        try {
            bytesRead = file.read(header, 0, HEADER_BYTES);
        } catch (TskCoreException ex) {
            return false; // can't read the first few bytes: don't parse
        }
        if (bytesRead <= 0) {
            return false;
        }

        // Signatures, most likely first. See
        // http://www.garykessler.net/library/file_sigs.html
        return startsWith(header, bytesRead, 0xFF, 0xD8, 0xFF)                              // JPEG
                || startsWith(header, bytesRead, 0x42, 0x4D)                                // BMP "BM"
                || startsWith(header, bytesRead, 0x89, 0x50, 0x4E, 0x47)                    // PNG (rest not checked)
                || startsWith(header, bytesRead, 0x47, 0x49, 0x46, 0x38)                    // GIF87a / GIF89a
                || startsWith(header, bytesRead, 0x49, 0x20, 0x49)                          // TIFF "I I"
                || startsWith(header, bytesRead, 0x49, 0x49, 0x2A, 0x00)                    // TIFF little endian
                || startsWith(header, bytesRead, 0x4D, 0x4D, 0x00, 0x2A)                    // TIFF big endian
                || startsWith(header, bytesRead, 0x4D, 0x4D, 0x00, 0x2B)                    // BigTIFF
                || startsWith(header, bytesRead, 0x00, 0x00, 0x00, 0x0C, 0x6A, 0x50, 0x20, 0x20); // JPEG 2000 (JP2)
    }

    /** @return true if the first sig.length bytes read match the signature */
    private static boolean startsWith(byte[] buf, int bytesRead, int... sig) {
        if (bytesRead < sig.length) {
            return false;
        }
        for (int i = 0; i < sig.length; i++) {
            if ((buf[i] & 0xFF) != sig[i]) {
                return false;
            }
        }
        return true;
    }

    /** Reads labels from model.names or model.txt next to the model, if present. */
    private static List<String> loadLabels(File modelFile) {
        String path = modelFile.getAbsolutePath();
        int dot = path.lastIndexOf('.');
        String base = dot > 0 ? path.substring(0, dot) : path;
        for (String ext : new String[]{".names", ".txt"}) {
            File f = new File(base + ext);
            if (f.isFile()) {
                try {
                    List<String> labels = new ArrayList<>();
                    for (String line : Files.readAllLines(f.toPath(), StandardCharsets.UTF_8)) {
                        if (!line.trim().isEmpty()) {
                            labels.add(line.trim());
                        }
                    }
                    return labels;
                } catch (IOException ex) {
                    logger.log(Level.WARNING, "Unable to read class names file " + f, ex);
                }
            }
        }
        return new ArrayList<>();
    }

    @Messages({"# {0} - detectionCount", "# {1} - summary",
        "ObjectDetectorOnnxFileIngestModule.classifierDetection.text=Model detected {0} object(s): {1}"})
    @Override
    public ProcessResult process(AbstractFile file) {
        if (models.isEmpty()) {
            return IngestModule.ProcessResult.OK;
        }

        // Skip anything other than actual file system files.
        if (file.getType() == TskData.TSK_DB_FILES_TYPE_ENUM.UNALLOC_BLOCKS
                || file.getType() == TskData.TSK_DB_FILES_TYPE_ENUM.UNUSED_BLOCKS
                || !file.isFile()) {
            return IngestModule.ProcessResult.OK;
        }

        // Skip unsupported file types.
        if (!isImageFile(file)) {
            return IngestModule.ProcessResult.OK;
        }

        if (file.getSize() > MAX_FILE_SIZE) {
            //prevent it from allocating gigabytes of memory for extremely large files
            logger.log(Level.INFO, "Encountered file " + file.getParentPath() + file.getName() + " with object id of "
                    + file.getId() + " which exceeds max file size of " + MAX_FILE_SIZE + " bytes, with a size of " + file.getSize());
            return IngestModule.ProcessResult.OK;
        }

        byte[] imageInMemory = new byte[(int) file.getSize()];
        try {
            file.read(imageInMemory, 0, file.getSize());
        } catch (TskCoreException ex) {
            logger.log(Level.WARNING, "Unable to read image to byte array for performing object detection on " + file.getParentPath() + file.getName() + " with object id of " + file.getId(), ex);
            return IngestModule.ProcessResult.ERROR;
        }

        Mat image = null;
        MatOfByte encoded = new MatOfByte(imageInMemory);
        try {
            image = Imgcodecs.imdecode(encoded, Imgcodecs.IMREAD_COLOR);
            if (image.empty()) {
                logger.log(Level.WARNING, "OpenCV could not decode " + file.getParentPath() + file.getName() + " with object id of " + file.getId());
                return IngestModule.ProcessResult.OK;
            }

            for (Map.Entry<String, OnnxModel> entry : models.entrySet()) {
                Map<String, Integer> counts;
                try {
                    counts = detect(entry.getValue(), image);
                } catch (CvException ex) {
                    logger.log(Level.WARNING, "Model " + entry.getKey() + " failed on " + file.getParentPath() + file.getName(), ex);
                    continue;
                } catch (Exception unexpectedException) {
                    //catch any undocumented exceptions OpenCv may throw
                    logger.log(Level.SEVERE, "Unexpected Exception for image " + file.getParentPath() + file.getName() + " with object id of " + file.getId() + " while applying model " + entry.getKey(), unexpectedException);
                    continue;
                }

                if (counts.isEmpty()) {
                    continue;
                }

                int total = 0;
                StringBuilder summary = new StringBuilder();
                for (Map.Entry<String, Integer> c : counts.entrySet()) {
                    total += c.getValue();
                    if (summary.length() > 0) {
                        summary.append(", ");
                    }
                    summary.append(c.getKey()).append(" (").append(c.getValue()).append(")");
                }

                try {
                    List<BlackboardAttribute> attributes = Arrays.asList(
                            new BlackboardAttribute(BlackboardAttribute.ATTRIBUTE_TYPE.TSK_DESCRIPTION, MODULE_NAME, entry.getKey()),
                            new BlackboardAttribute(BlackboardAttribute.ATTRIBUTE_TYPE.TSK_COMMENT, MODULE_NAME,
                                    Bundle.ObjectDetectorOnnxFileIngestModule_classifierDetection_text(total, summary.toString()))
                    );

                    BlackboardArtifact artifact = file.newAnalysisResult(
                            BlackboardArtifact.Type.TSK_OBJECT_DETECTED, Score.SCORE_UNKNOWN, null, null, null, attributes)
                            .getAnalysisResult();

                    try {
                        /*
                         * Index the artifact for keyword search.
                         */
                        blackboard.postArtifact(artifact, MODULE_NAME, jobId);
                    } catch (Blackboard.BlackboardException ex) {
                        logger.log(Level.SEVERE, "Unable to index blackboard artifact " + artifact.getArtifactID(), ex); //NON-NLS
                    }
                } catch (TskCoreException ex) {
                    logger.log(Level.SEVERE, String.format("Failed to create blackboard artifact for '%s'.", file.getParentPath() + file.getName()), ex); //NON-NLS
                    return IngestModule.ProcessResult.ERROR;
                }
            }
        } catch (CvException ex) {
            logger.log(Level.WARNING, "Unable to decode image from byte array to perform object detection on " + file.getParentPath() + file.getName() + " with object id of " + file.getId(), ex); //NON-NLS
            return IngestModule.ProcessResult.ERROR;
        } catch (Exception unexpectedException) {
            logger.log(Level.SEVERE, "Unexpected Exception encountered attempting to use OpenCV to decode picture: " + file.getParentPath() + file.getName() + " with object id of " + file.getId(), unexpectedException);
            return IngestModule.ProcessResult.ERROR;
        } finally {
            encoded.release();
            if (image != null) {
                image.release();
            }
        }

        return IngestModule.ProcessResult.OK;
    }

    /**
     * Runs one ONNX model on a BGR image and returns detection counts per
     * label (after confidence filtering and NMS).
     *
     * Supports YOLOv8/v11-style output [1, 4+nc, N] and YOLOv5-style output
     * [1, N, 5+nc] (the latter is recognised when the label count is known).
     */
    private static Map<String, Integer> detect(OnnxModel model, Mat image) {
        Map<String, Integer> counts = new LinkedHashMap<>();

        // Letterbox to the model's input size, preserving aspect ratio.
        final int inW = model.inputWidth;
        final int inH = model.inputHeight;
        int w = image.cols();
        int h = image.rows();
        double scale = Math.min((double) inW / w, (double) inH / h);
        int newW = Math.max(1, (int) Math.round(w * scale));
        int newH = Math.max(1, (int) Math.round(h * scale));
        int padX = (inW - newW) / 2;
        int padY = (inH - newH) / 2;

        Mat resized = new Mat();
        Mat padded = new Mat();
        Mat blob = new Mat();
        List<Mat> outputs = new ArrayList<>();
        Mat raw = null;
        Mat flat = null;
        Mat rows2d = null;
        MatOfRect2d boxMat = new MatOfRect2d();
        MatOfFloat scoreMat = new MatOfFloat();
        MatOfInt keep = new MatOfInt();
        try {
            Imgproc.resize(image, resized, new Size(newW, newH));
            Core.copyMakeBorder(resized, padded, padY, inH - newH - padY, padX, inW - newW - padX,
                    Core.BORDER_CONSTANT, new Scalar(114, 114, 114));
            blob = Dnn.blobFromImage(padded, 1.0 / 255.0, new Size(padded.cols(), padded.rows()), new Scalar(0, 0, 0), true, false);

            model.net.setInput(blob);
            model.net.forward(outputs, model.net.getUnconnectedOutLayersNames());
            if (outputs.isEmpty()) {
                return counts;
            }
            raw = outputs.get(0);
            if (raw.dims() != 3) {
                return counts;
            }

            int d1 = raw.size(1);
            int d2 = raw.size(2);
            flat = raw.reshape(1, d1); // d1 x d2
            boolean channelsFirst = d1 < d2;
            int numBoxes = channelsFirst ? d2 : d1;
            int cols = channelsFirst ? d1 : d2;

            float[] data = new float[d1 * d2];
            flat.get(0, 0, data);

            boolean hasObjectness = !model.labels.isEmpty() && cols == model.labels.size() + 5;
            int classStart = hasObjectness ? 5 : 4;
            int numClasses = cols - classStart;
            if (numClasses <= 0) {
                return counts;
            }

            List<Rect2d> boxes = new ArrayList<>();
            List<Float> scores = new ArrayList<>();
            List<Integer> classIds = new ArrayList<>();

            for (int i = 0; i < numBoxes; i++) {
                float cx = value(data, channelsFirst, i, 0, d2, cols);
                float cy = value(data, channelsFirst, i, 1, d2, cols);
                float bw = value(data, channelsFirst, i, 2, d2, cols);
                float bh = value(data, channelsFirst, i, 3, d2, cols);
                float objectness = hasObjectness ? value(data, channelsFirst, i, 4, d2, cols) : 1f;

                int bestClass = -1;
                float bestScore = 0f;
                for (int c = 0; c < numClasses; c++) {
                    float s = value(data, channelsFirst, i, classStart + c, d2, cols);
                    if (s > bestScore) {
                        bestScore = s;
                        bestClass = c;
                    }
                }
                float conf = bestScore * objectness;
                if (bestClass < 0 || conf < CONFIDENCE_THRESHOLD) {
                    continue;
                }

                // Map back from letterboxed coordinates to the original image.
                double x = (cx - bw / 2.0 - padX) / scale;
                double y = (cy - bh / 2.0 - padY) / scale;
                boxes.add(new Rect2d(x, y, bw / scale, bh / scale));
                scores.add(conf);
                classIds.add(bestClass);
            }

            if (boxes.isEmpty()) {
                return counts;
            }

            float[] scoreArr = new float[scores.size()];
            for (int i = 0; i < scoreArr.length; i++) {
                scoreArr[i] = scores.get(i);
            }
            boxMat.fromList(boxes);
            scoreMat.fromArray(scoreArr);
            Dnn.NMSBoxes(boxMat, scoreMat, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, keep);

            for (int idx : keep.toArray()) {
                int classId = classIds.get(idx);
                String label = classId < model.labels.size() ? model.labels.get(classId) : "class " + classId;
                counts.merge(label, 1, Integer::sum);
            }
            return counts;
        } finally {
            resized.release();
            padded.release();
            blob.release();
            for (Mat m : outputs) {
                m.release();
            }
            if (flat != null) {
                flat.release();
            }
            if (rows2d != null) {
                rows2d.release();
            }
            boxMat.release();
            scoreMat.release();
            keep.release();
        }
    }

    /** Reads element (box i, field f) from a [fields x boxes] or [boxes x fields] flat array. */
    private static float value(float[] data, boolean channelsFirst, int box, int field, int d2, int cols) {
        return channelsFirst ? data[field * d2 + box] : data[box * cols + field];
    }

    @Override
    public void shutDown() {
        refCounter.decrementAndGet(jobId);
        if (models != null) {
            models.clear();
        }
    }
}