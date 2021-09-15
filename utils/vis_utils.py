import cv2

def draw_tags(img, tags):
    canvas = img.copy()
    for tag in tags:
        for idx in range(len(tag.corners)):
            cv2.line(
                canvas,
                tuple(tag.corners[idx - 1, :].astype(int)),
                tuple(tag.corners[idx, :].astype(int)),
                (0, 200, 0), 2
            )
            cv2.putText(
                canvas,
                str(idx),
                tuple(tag.corners[idx, :].astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            canvas,
            str(tag.tag_id),
            (tag.center[0].astype(int) - 20,
             tag.center[1].astype(int) + 20,),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    return canvas

def draw_points(img, points):
    canvas = img.copy()
    for point in points:
        cv2.circle(canvas, (int(point[0]), int(point[1])), 2, (0, 255, 0), 1, cv2.LINE_AA)
    return canvas