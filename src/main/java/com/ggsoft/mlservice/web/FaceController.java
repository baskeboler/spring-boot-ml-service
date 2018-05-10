package com.ggsoft.mlservice.web;

import com.ggsoft.mlservice.domain.Face;
import com.ggsoft.mlservice.services.FaceService;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;

import javax.annotation.PostConstruct;
import java.util.List;

import static org.springframework.http.MediaType.APPLICATION_JSON_UTF8_VALUE;
import static org.springframework.http.MediaType.APPLICATION_OCTET_STREAM_VALUE;

@RestController
@Slf4j
@RequestMapping("/api")
public class FaceController {


    private FaceService faceService;

    @Autowired
    public FaceController(FaceService faceService) {
        this.faceService = faceService;
    }

    @PostConstruct
    public void init() {
        log.info("Initializing FaceController");
    }


    @PostMapping(value = "/face",
        produces = APPLICATION_JSON_UTF8_VALUE,
        consumes = APPLICATION_OCTET_STREAM_VALUE)
    public @ResponseBody
    List<Face> detect(@RequestBody byte[] image) {
        MatOfByte mat = new MatOfByte();
        mat.fromArray(image);
        Mat decoded = Imgcodecs.imdecode(mat, Imgcodecs.IMREAD_GRAYSCALE);
        List<Face> faces = faceService.detect(decoded);
        return faces;
    }

    @PostMapping(value = "/face/{label}/train",
        produces = APPLICATION_JSON_UTF8_VALUE,
        consumes = APPLICATION_OCTET_STREAM_VALUE)
    public void detect(@RequestBody byte[] image, @PathVariable int label) {
        MatOfByte mat = new MatOfByte();
        mat.fromArray(image);
        Mat decoded = Imgcodecs.imdecode(mat, Imgcodecs.IMREAD_GRAYSCALE);
        faceService.train(decoded, label);
    }


    @PostMapping(value = "/face/predict",
        produces = APPLICATION_JSON_UTF8_VALUE,
        consumes = APPLICATION_OCTET_STREAM_VALUE)
    public @ResponseBody
    FaceService.PredictionResult predict(@RequestBody byte[] image) {
        MatOfByte mat = new MatOfByte();
        mat.fromArray(image);
        Mat decoded = Imgcodecs.imdecode(mat, Imgcodecs.IMREAD_GRAYSCALE);
        return faceService.predict(decoded);
    }
}
