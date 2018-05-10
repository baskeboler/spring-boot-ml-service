package com.ggsoft.mlservice.config;

import lombok.extern.slf4j.Slf4j;
import org.opencv.objdetect.CascadeClassifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.annotation.Order;

@Configuration
@Order(0)
@Slf4j
public class OpenCVConfig {


    @Value("${ml-service.opencv.cascades.face.path}")
    private String faceCascadePath;


    public CascadeClassifier faceCascade() {
        log.info("Initializing new face cascade");
        CascadeClassifier cascade = new CascadeClassifier();
        log.info("Loading cascade from {}", faceCascadePath);
        boolean load = cascade.load(faceCascadePath);
        if (!load) {
            throw new RuntimeException("Failed to load cascade");
        }
        log.info("cascade initialized");
        return cascade;
    }
}
