package com.ggsoft.mlservice.services;

import com.ggsoft.mlservice.config.OpenCVConfig;
import com.ggsoft.mlservice.domain.Face;
import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.face.LBPHFaceRecognizer;
import org.opencv.face.PredictCollector;
import org.opencv.face.StandardCollector;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Collections;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
@Slf4j
public class FaceService {

    @Autowired
    private OpenCVConfig config;

    private LBPHFaceRecognizer faceRecognizer;

    private CascadeClassifier faceCascadeClassifier;

    @PostConstruct
    public void init() throws IOException {
        log.info("Initializing FaceService");
        faceCascadeClassifier = config.faceCascade();

        faceRecognizer = LBPHFaceRecognizer.create();

        trainDir("/home/victor/tmp/people/victor", 1);
        trainDir("/home/victor/tmp/people/leti", 2);
        trainDir("/home/victor/tmp/people/vicky", 3);
    }


    public List<Face> detect(Mat image) {
        Mat greyscale = image;
        MatOfRect result = new MatOfRect();
        log.info("Detecting faces");
        faceCascadeClassifier.detectMultiScale(
            greyscale,
            result,
            1.1,
            2,
            Objdetect.CASCADE_SCALE_IMAGE,
            new Size(50, 50),
            greyscale.size());
        log.info("Got back {} faces", result.total());

        return result.toList()
            .stream()
            .map(Face::fromRect)
            .collect(Collectors.toList());
    }

    public void train(Mat image, int label) {
        List<Mat> histograms = faceRecognizer.getHistograms();
        if (histograms.isEmpty()) {
            faceRecognizer.train(Collections.singletonList(image), new MatOfInt(label));
        } else {

            faceRecognizer.update(Collections.singletonList(image), new MatOfInt(label));
        }
    }

    public PredictionResult predict(Mat image) {
        PredictCollector collector = StandardCollector.create();

        faceRecognizer.predict_collect(image, collector);
        return PredictionResult.builder()
            .label(((StandardCollector) collector).getMinLabel())
            .confidence(((StandardCollector) collector).getMinDist())
            .build();
    }

    private void getFaces(String src, String dst) throws IOException {
        int faceCount = 0;
        Files.list(new File(src).toPath())
            .forEach(f -> {
                Mat img = Imgcodecs.imread(f.toAbsolutePath().toString());
                List<Face> detect = detect(img);
                detect.stream()
                    .forEach(face -> {
                        String uuid = UUID.randomUUID().toString();
                        Mat faceImg = img.submat(new Rect(face.getX(), face.getY(), face.getWidth(), face.getHeight()));
                        boolean imwrite = Imgcodecs.imwrite(String.format("%s/%s.jpg", dst, uuid), faceImg);
                    });
            });
    }

    private void trainDir(String path, int label) throws IOException {
        log.info("Training dir {} for label {}", path, label);
        Files.list(new File(path).toPath())
            .forEach(f -> {
                Mat img = Imgcodecs.imread(f.toAbsolutePath().toString(), Imgcodecs.IMREAD_GRAYSCALE);
                train(img, label);
            });
    }

    @Data
    @Builder
    public static class PredictionResult {
        int label;
        double confidence;
    }
}
