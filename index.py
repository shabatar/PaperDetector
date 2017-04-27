from RGBHistogram import RGBHistogram
import argparse
import pickle
import glob
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the catalog that contains the images to be indexed")
ap.add_argument("-i", "--index", required=True, help="Path to index")
args = vars(ap.parse_args())

datasetDir = args["dataset"]

index = {}
desc = RGBHistogram([8, 8, 8])

INIT_SIZE = 800

step = 0

for imagePath in glob.glob(datasetDir + "/*.*g"):
    photoName = imagePath[imagePath.rfind("/") + 1:]

    image = cv2.imread(imagePath)

    size = image.shape[:2]
    coefficient = size[0] / size[1]
    # Resize image
    width = INIT_SIZE
    height = int(INIT_SIZE * coefficient)
    image = cv2.resize(image, (width, height))
    # Crop center
    image = image[height // 8: (7 * height) // 8, width // 8: (7 * width) // 8]

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    features = desc.describe(image)
    index[photoName] = features

    if step % 20 == 0:
        print("progress: %d" % step)
    step += 1

f = open(args["index"], "wb")
f.write(pickle.dumps(index))
f.close()

print("Done. Indexed %d images" % (len(index)))
