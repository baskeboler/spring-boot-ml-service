package com.ggsoft.mlservice;

import com.ggsoft.mlservice.config.OpenCVConfig;
import com.ggsoft.mlservice.config.WebConfig;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.Core;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Import;

@Slf4j
@EnableAutoConfiguration
@SpringBootApplication
@Import( {OpenCVConfig.class, WebConfig.class})
public class MlServiceApplication {

    //    static {
//        log.info("This is the library path: {}", System.getProperty("java.library.path"));
//        Library.
//        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//        log.info("Library loaded");
//    }
    public static void main(String[] args) {
        log.info("Loading OpenCV native library");
        log.info("This is the library path: {}", System.getProperty("java.library.path"));
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        System.loadLibrary("face_recognizer");
        log.info("Library loaded");
//        String oldPath = System.getProperty("java.library.path");
//        String openCVPath = "/home/victor/.local/share/OpenCV/java";
//        String localLibs = "/home/victor/.local/lib:/usr/local/lib";
//        System.setProperty("java.library.path", String.format("%s:%s:%s", openCVPath, oldPath, localLibs));
        SpringApplication.run(MlServiceApplication.class, args);
    }


}
