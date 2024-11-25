import cv2 as cv
import numpy as np
import numpy.linalg as la
from scipy.ndimage import gaussian_filter

def kalman(mu,P,F,Q,B,u,z,H,R):
    # mu, P : estado actual y su incertidumbre
    # F, Q  : sistema dinámico y su ruido
    # B, u  : control model y la entrada
    # z     : observación
    # H, R  : modelo de observación y su ruido
    
    mup = F @ mu + B @ u
    pp  = F @ P @ F.T + Q

    zp = H @ mup

    # si no hay observación solo hacemos predicción 
    if z is None:
        return mup, pp, zp

    epsilon = z - zp

    k = pp @ H.T @ la.inv(H @ pp @ H.T +R)

    new_mu = mup + k @ epsilon
    new_P  = (np.eye(len(P))-k @ H) @ pp
    return new_mu, new_P, zp

REDU = 8

def rgbh(xs, mask):
	def normhist(x): 
		return x / np.sum(x)
	
	def h(rgb):
		return cv.calcHist([rgb], [0, 1, 2], mask, [256 // REDU, 256 // REDU, 256 // REDU], [0, 256] + [0, 256] + [0, 256])
	
	return normhist(sum(map(h, xs)))

def smooth(s, x):
	return gaussian_filter(x, s, mode='constant')

bgsub = cv.createBackgroundSubtractorMOG2(500, 60, True)  # The threshold value could vary (60)

key = 0

termination = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

###################### Initial Kalman ########################
# state that Kalman is updating. This is the initial value
degree = np.pi / 180
a = np.array([0, 900])

fps = 100

dt = 1 / fps
t = np.arange(0, 20, dt)
noise = 1

F = np.array([
	1, 0, dt, 0,
	0, 1, 0, dt,
	0, 0, 1, 0,
	0, 0, 0, 1
]).reshape(4, 4)

B = np.array([
	dt**2 / 2, 0,
	0, dt**2 / 2,
	dt, 0,
	0, dt
]).reshape(4, 2)

H = np.array([
	1, 0, 0, 0,
	0, 1, 0, 0
]).reshape(2, 4)


sigmaM = 0.0001  # model noise
sigmaZ = 3 * noise  # should be equal to the average noise of the image process. 10 pixels approx.

Q = sigmaM**2 * np.eye(4)
R = sigmaZ**2 * np.eye(2)

# Initialize variables
crop = False
pause = False
camshift = False
mu = np.array([0, 0, 0, 0])  # Initial state vector for Kalman filter
P = np.diag([100, 100, 100, 100])**2  # Initial covariance matrix
list_center_x = []  # List to store x coordinates of the object's center
list_center_y = []  # List to store y coordinates of the object's center
list_points = []  # List to store points and status
res = []  # List to store Kalman filter predictions
font = cv.FONT_HERSHEY_SIMPLEX  # Font for text in frames
kernel = np.ones((3, 3), np.uint8)  # Kernel for morphological operations

# Start video capture
cap = cv.VideoCapture("projectile_motion.mov")

while True:
    # Capture keyboard input
    key = cv.waitKey(1) & 0xFF
    if key == ord("c"):
        crop = True  # Enable cropping
    if key == ord("p"):
        P = np.diag([100, 100, 100, 100])**2  # Reset covariance matrix
    if key == 27:  # Press ESC to exit
        break
    if key == ord(" "):  # Toggle pause/resume
        pause = not pause
    
    if pause:
        continue  # Skip frame processing if paused

    # Read the current frame
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    bgs = bgsub.apply(frame)
    bgs = cv.erode(bgs, kernel, iterations=1)  # Erode noise
    bgs = cv.medianBlur(bgs, 3)  # Apply median blur
    bgs = cv.dilate(bgs, kernel, iterations=2)  # Dilate foreground
    bgs = (bgs > 100).astype(np.uint8) * 255  # Threshold the background mask

    # Apply the mask to the frame
    color_mask = cv.bitwise_and(frame, frame, mask=bgs)

    if crop:
        # Enable ROI selection for cropping
        from_center = False
        img = color_mask
        roi = cv.selectROI(img, from_center)  # Select region of interest (ROI)
        cropped_img = img[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        crop = False  # Disable cropping after selection
        camshift = True  # Enable CamShift tracking
        cropped_img_mask = cv.cvtColor(cropped_img, cv.COLOR_BGR2GRAY)
        ret, cropped_img_mask = cv.threshold(cropped_img_mask, 30, 255, cv.THRESH_BINARY)
        hist = smooth(1, rgbh([cropped_img], cropped_img_mask))  # Calculate histogram

        roi_box = (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
        cv.destroyWindow("ROI selector")  # Close the ROI selector window

    if camshift:
        # Display text for tracking and prediction
        cv.putText(frame, 'Center roiBox', (0, 10), font, 0.5, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(frame, 'Estimated position', (0, 30), font, 0.5, (255, 255, 0), 2, cv.LINE_AA)
        cv.putText(frame, 'Prediction', (0, 50), font, 0.5, (0, 0, 255), 2, cv.LINE_AA)

        # Backprojection for color model
        rgbr = np.floor_divide(color_mask, REDU)
        r, g, b = rgbr.transpose(2, 0, 1)
        l = hist[r, g, b]
        max_l = l.max()
        back_projection = np.clip((1 * l / max_l * 255), 0, 255).astype(np.uint8)
        # Uncomment to view backprojection
        # cv.imshow("Backprojection", cv.resize(back_projection, (400, 250)))

        # Apply CamShift and draw ellipse
        (rb, roi_box) = cv.CamShift(l, roi_box, termination)
        cv.ellipse(frame, rb, (0, 255, 0), 2)

        ########## Kalman filter ############
        # Calculate object center coordinates
        x_center = int(roi_box[0] + roi_box[2] / 2)
        y_center = int(roi_box[1] + roi_box[3] / 2)
        error_threshold = roi_box[3]  # Threshold for determining valid movement
        
        if y_center < error_threshold or bgs.sum() < 50:
            # If object is too small or background is too noisy, predict using Kalman filter
            mu, P, pred = kalman(mu, P, F, Q, B, a, None, H, R)
            state = "None"
            is_moving = False
        else:
            # If object movement is valid, update Kalman filter with new observation
            mu, P, pred = kalman(mu, P, F, Q, B, a, np.array([x_center, y_center]), H, R)
            state = "normal"
            is_moving = True

        if is_moving:
            # Store the tracked position and Kalman filter result
            list_center_x.append(x_center)
            list_center_y.append(y_center)
        list_points.append((x_center, y_center, state))
        res.append((mu, P))

        ##### Prediction #####
        # Predict future positions using Kalman filter
        mu2 = mu
        P2 = P
        res2 = []

        for _ in range(fps): 
            mu2, P2, pred2 = kalman(mu2, P2, F, Q, B, a, None, H, R)
            res2.append((mu2, P2))

        # Extract Kalman filter results for plotting
        x_estimated = [mu[0] for mu, _ in res]
        x_uncertainty = [2 * np.sqrt(P[0, 0]) for _, P in res]
        y_estimated = [mu[1] for mu, _ in res]
        y_uncertainty = [2 * np.sqrt(P[1, 1]) for _, P in res]

        x_predicted = [mu2[0] for mu2, _ in res2]
        y_predicted = [mu2[1] for mu2, _ in res2]
        x_predicted_uncertainty = [2 * np.sqrt(P2[0, 0]) for _, P2 in res2]
        y_predicted_uncertainty = [2 * np.sqrt(P2[1, 1]) for _, P2 in res2]

        # Draw the tracked positions and uncertainties on the frame
        for n in range(len(list_center_x)):
            cv.circle(frame, (int(list_center_x[n]), int(list_center_y[n])), 3, (0, 255, 0), -1)

        for n in [-1]:  # For the last position in the list
            uncertainty = (x_uncertainty[n] + y_uncertainty[n]) / 2
            cv.circle(frame, (int(x_estimated[n]), int(y_estimated[n])), int(uncertainty), (255, 255, 0), 1)

        for n in range(len(x_predicted)):  # Predicted positions
            uncertainty_predicted = (x_predicted_uncertainty[n] + y_predicted_uncertainty[n]) / 2
            cv.circle(frame, (int(x_predicted[n]), int(y_predicted[n])), int(uncertainty_predicted), (0, 0, 255))


        # # Detect rebound (for bouncing objects)
        # if len(list_center_y) > 4:
        #     if (list_center_y[-5] < list_center_y[-4]) and (list_center_y[-4] < list_center_y[-3]) and (list_center_y[-3] > list_center_y[-2]) and (list_center_y[-2] > list_center_y[-1]):
        #         print("REBOTE")  # Rebound detected
        #         # Reset lists and Kalman filter for new tracking
        #         list_center_y = []
        #         list_center_x = []
        #         list_points = []
        #         res = []
        #         mu = np.array([0, 0, 0, 0])
        #         P = np.diag([100, 100, 100, 100])**2

    # Display frames
    cv.imshow('ColorMask', color_mask)
    cv.imshow('Mask', bgs)
    cv.imshow('Frame', frame)

cap.release()
