from RGBHistogram import RGBHistogram
from searcher import Searcher
import argparse
import pickle
import cv2
import glob

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--index", required=True)
ap.add_argument("-q", "--query", required=True)
ap.add_argument("-t", "--type", required=True)
args = vars(ap.parse_args())

queryDir = args["query"]
isOriginalPapers = args["type"] == "paper"

# Assume that given photos are papers IFF the directory contains "not" or "Not"
#isOriginalPapers = (queryDir.find("not") < 0 and queryDir.find("Not") < 0)
if isOriginalPapers:
    print("This program prints the names of photos detected as NOT PAPER")
else:
    print("This program prints the names of photos detected as PAPER")
print()

desc = RGBHistogram([8, 8, 8])
index = pickle.loads(open(args["index"], "rb").read())
searcher = Searcher(index)

# Number of photos detected as "paper" or "not paper"
isPaperCnt = 0
isNotPaperCnt = 0

# Size of photos
INIT_SIZE = 800

# Get all .jpg and .png files from directory
for imagePath in glob.glob(queryDir + "/*.*g"):
    photoName = imagePath[imagePath.rfind("/") + 1:]

    image = cv2.imread(imagePath)

    size = image.shape[:2]
    coefficient = size[0] / size[1]
    # Resize image
    width = INIT_SIZE
    height = int(INIT_SIZE * coefficient)
    image = cv2.resize(image, (width, height))
    # Crop center
    imageCenter = image[height // 8: (7 * height) // 8, width // 8: (7 * width) // 8]
    imageSmallCenter = image[height // 4: (3 * height) // 4, width // 4: (3 * width) // 4]

    image = imageCenter

    gray = cv2.cvtColor(imageSmallCenter, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    imageFeatures = desc.describe(image)
    results = searcher.search(imageFeatures)

    # Number of similar images from index
    samePapers = 0
    sameNotPapers = 0

    # Get top 10 of same images
    for j in range(0, 10):
        (score, imageName) = results[j]
        # print("\t%d. %s : %.3f" % (j + 1, imageName, score))

        if imageName[:3] == "not":
            sameNotPapers += 1
        else:
            samePapers += 1

    # If the number of similar paper-images is larger than number of similar not-paper-images,
    # it's very likely that the photo is paper-image
    if samePapers > 5:
        isPaperCnt += 1

        # Paper-images usually contain about 1-5 contours
        if len(contours) > 6:
            print(photoName + ": too many contours => it's not a paper!" + " (" + str(len(contours)) + ")")
            isPaperCnt -= 1
            isNotPaperCnt += 1

        if (not isOriginalPapers):
            print(photoName + ": is paper (is = %d, not = %d, cnts = %d)" % (samePapers, sameNotPapers, len(contours)))
            '''
            for j in range(0, 10):
                (score, imageName) = results[j]
                print(imageName, score)
            '''
            print()
    else:
        isNotPaperCnt += 1
        if (isOriginalPapers):
            print(photoName + ": is not paper (is = %d, not = %d, cnts = %d)" % (samePapers, sameNotPapers, len(contours)))
            '''
            for j in range(0, 10):
                (score, imageName) = results[j]
                print(imageName, score)
            '''
            print()

print()
if not isOriginalPapers:
    print("Correct: %d, mistake: %d" % (isNotPaperCnt, isPaperCnt))
else:
    print("Correct: %d, mistake: %d" % (isPaperCnt, isNotPaperCnt))
