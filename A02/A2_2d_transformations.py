import numpy as np
import torch
import cv2

SMILE_PATH = "./smile.png"

img_smile = cv2.imread(SMILE_PATH)

def warp_image(img, H):
    x, y = np.meshgrid(
        np.linspace(
            -400,
            400,
            801
        ),
        np.linspace(
            -400,
            400,
            801
        )
    )
    coords = np.stack([
        x.reshape(-1),
        y.reshape(-1),
        np.ones(x.reshape(-1).shape)
    ])

    coords_transformed = np.matmul(
        np.linalg.inv(H),
        coords
    )
    coords_transformed /= coords_transformed[-1]

    coords_transformed[:2, :] += 50 # 좌표걔 맞추기

    def sampleFromImage(coord) :
        # coord : [x, y]
        if 0 <= coord[0] < img.shape[1] and 0 <= coord[1] < img.shape[0] :
            return img[coord[1], coord[0], :]
        else :
            return np.array([255] * 3)

    img_transformed = np.array(
        list(map(
            sampleFromImage,
            coords_transformed[:2].T.astype(np.uint32)
        ))
    ).reshape(801, 801, 3).astype(np.uint8)
    
    cv2.arrowedLine(
        img_transformed,
        (0, 400),
        (800, 400),
        (0, 0, 0),
        2
    )
    cv2.arrowedLine(
        img_transformed,
        (400, 0),
        (400, 800),
        (0, 0, 0),
        2
    )
    
    return img_transformed



class ImageWarper :

    image = img_smile
    H = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    instructions = {
        'a' : np.array([
            [1, 0, -5],
            [0, 1, 0],
            [0, 0, 1]
        ]),
        'd' : np.array([
            [1, 0, 5],
            [0, 1, 0],
            [0, 0, 1]
        ]),
        'w' : np.array([
            [1, 0, 0],
            [0, 1, -5],
            [0, 0, 1]
        ]),
        's' : np.array([
            [1, 0, 0],
            [0, 1, 5],
            [0, 0, 1]
        ]),

        'r' : np.array([
            [ np.cos(np.deg2rad(5)), np.sin(np.deg2rad(5)), 0],
            [-np.sin(np.deg2rad(5)), np.cos(np.deg2rad(5)), 0],
            [0, 0, 1]
        ]),
        't' : np.array([
            [np.cos(np.deg2rad(-5)), np.sin(np.deg2rad(-5)), 0],
            [-np.sin(np.deg2rad(-5)), np.cos(np.deg2rad(-5)), 0],
            [0, 0, 1]
        ]),

        'f' : np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]),
        'g' : np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ]),
        
        'x' : np.array([
            [0.95, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]),
        'c' : np.array([
            [1.05, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]),
        'y' : np.array([
            [1, 0, 0],
            [0, 0.95, 0],
            [0, 0, 1]
        ]),
        'u' :np.array([
            [1, 0, 0],
            [0, 1.05, 0],
            [0, 0, 1]
        ]),
    }

    def start() :
        while True :

            print(ImageWarper.H)

            cv2.imshow(
                "image",
                warp_image(
                    ImageWarper.image,
                    ImageWarper.H
                )
            )

            key = cv2.waitKey(-1)

            print("got key :", chr(key))

            if key == ord("q") :
                cv2.destroyAllWindows()
                break
            elif key == ord("h") :
                ImageWarper.H = np.array([
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]
                ])
            elif chr(key) in ImageWarper.instructions.keys() :
                ImageWarper.H = np.matmul(
                    ImageWarper.instructions[chr(key)],
                    ImageWarper.H,
                )
            else :
                print("invalid instruction")


if __name__ == "__main__" :
    ImageWarper.start()
