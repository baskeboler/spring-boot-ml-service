package com.ggsoft.mlservice.domain;

import lombok.Builder;
import lombok.Data;
import org.opencv.core.Rect;

@Data
@Builder
public class Face {

    private int x, y, width, height;

    public static final Face fromRect(Rect r) {
        return Face.builder()
            .x(r.x)
            .y(r.y)
            .width(r.width)
            .height(r.height)
            .build();
    }
}
