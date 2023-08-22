import numpy as np
import cv2 as cv
import glob
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((7*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.pgm')

for fname in images:
    img = cv.imread(fname, 0)

    ret, corners = cv.findChessboardCorners(img, (9,7), None)
    if ret == True:
        objpoints.append(objp)

    corners2 = cv.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)

    imgpoints.append(corners2)

    #cv.drawChessboardCorners(img, (9,7), corners2, ret)
    #cv.imshow('img', img)
    #cv.waitKey(500)

cv.destroyAllWindows()


# Calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)

# Undistort
for fname in images:
    img = cv.imread(fname, 0)

    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))

    # --- Method 2 ---
    #dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    # --- Method 2 ---
    # undistort
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite(fname + 'calibresult.png', dst)

# Save mapx, mapy
#with open("mapx.dat", "wb") as bfile:
#    bfile.write(mapx)
#with open("mapy.dat", "wb") as bfile:
#    bfile.write(mapy)


# Replace 'output_filename.h' with your desired output filename
output_filename = 'mapxpy.h'

# Open the file for writing
with open(output_filename, 'w') as f:
    f.write('#pragma once\n\n')
    f.write('#include <cstdint>\n\n')
    f.write('constexpr std::uint32_t arraySize = {};\n'.format(mapx.shape[0] * mapx.shape[1]))

    f.write('constexpr float mapx[arraySize] = {\n')
    # Write the array values
    for i in range(mapx.shape[0]):
        for j in range(mapx.shape[1]):
            f.write('        {},\n'.format(mapx[i,j]))
    f.write('    };\n')

    f.write('constexpr float mapy[arraySize] = {\n')
    # Write the array values
    for i in range(mapy.shape[0]):
        for j in range(mapy.shape[1]):
            f.write('        {},\n'.format(mapy[i,j]))
    f.write('    };\n')
