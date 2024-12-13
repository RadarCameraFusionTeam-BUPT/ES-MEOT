import argparse
import numpy as np
import cv2
import os

ROOT_DIR = os.path.abspath("./")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


def draw_box(image, box, color):
    """Draw 3-pixel width bounding boxes on the given image array.
    color: list of 3 int values for RGB.
    """
    y1, x1, y2, x2 = box
    image[y1:y1 + 2, x1:x2] = color
    image[y2:y2 + 2, x1:x2] = color
    image[y1:y2, x1:x1 + 2] = color
    image[y1:y2, x2:x2 + 2] = color
    return image


def ColorFrame(image, boxes, ids, class_names, scores=None):
    """
    image: numpy array.
    boxes: [num_instance, (x1, y1, x2, y2)] in image coordinates.
    ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == ids.shape[0]

    # Show area outside image boundaries.
    height, width = image.shape[:2]

    # masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        ID = ids[i]
        color = [0, 0, 0]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        x1, y1, x2, y2 = boxes[i]
        draw_box(image, [y1, x1, y2, x2], np.array(color)*255)

        # Label
        score = scores[i] if scores is not None else None
        label = class_names[i]
        lab = "{} {}".format(ID, label)
        sco = "{:.3f}".format(score)
        cv2.putText(image, lab, (x1, y1+15),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
        cv2.putText(image, sco, (x1, y1+15+23),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)

    return image


def Solve(args):
    ####### parameters ##########
    DATA_PATH = args.dataPath[0]

    if not os.path.exists('{}/output-keypoints.npy'.format(DATA_PATH)):
        print('{}/output-keypoints.npy not exist!'.format(DATA_PATH))
        exit(0)
    if not os.path.exists('{}/output.avi'.format(DATA_PATH)):
        print('{}/output.avi not exist!'.format(DATA_PATH))
        exit(0)

    ########## Load npy and video ###########
    npyData = np.load(
        '{}/output-keypoints.npy'.format(DATA_PATH), allow_pickle=True)

    
    cap = cv2.VideoCapture('{}/output.avi'.format(DATA_PATH))
    if not cap.isOpened():
        print('Open {}/output.avi failed!'.format(DATA_PATH))
        exit(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.save_vid:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        savePath = os.path.join(OUTPUT_DIR, 'output-keypoints-ByteTrack.mp4')
        out = cv2.VideoWriter(savePath, fourcc, 20.0, (width, height))

    # Load tracker
    from tracker.byte_tracker import BYTETracker
    tracker = BYTETracker(args)

    # Solve and save
    results = []
    for ID in range(len(npyData)):
        dets = np.column_stack((npyData[ID]['bbox'][:, 1], npyData[ID]['bbox'][:, 0],
                               npyData[ID]['bbox'][:, 3], npyData[ID]['bbox'][:, 2], npyData[ID]['score']))

        extraInfo = np.array([{'keypoints': npyData[ID]['keypoints'][i], 'category': npyData[ID]
                                ['category'][i]} for i in range(len(npyData[ID]['keypoints']))])

        now = dict()
        if len(dets) > 0:
            online_targets = tracker.update(
                dets, [height, width], [height, width], extraInfo)
        else:
            online_targets = []
        online_tlwhs = []
        online_ids = []
        online_scores = []
        online_category = []
        online_keypoints = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
                # print('>>>>>>>>>>>>>>>>>')
                # print(t.extraInfo)
                online_category.append(t.extraInfo['category'])
                online_keypoints.append(t.extraInfo['keypoints'])

        for item in online_tlwhs:
            item[2] += item[0]
            item[3] += item[1]
            item[0], item[1], item[2], item[3] = item[1], item[0], item[3], item[2]

        now['boxes'] = np.array(online_tlwhs).astype(int)
        now['ids'] = np.array(online_ids)
        now['scores'] = np.array(online_scores)
        now['category'] = np.array(online_category)
        now['keypoints'] = online_keypoints
        results.append(now)

        if args.save_vid:
            ret, frame = cap.read()
            if not ret:
                print('Error: the frame numbers of video and npy are different!!')
                break
            frameDis = ColorFrame(
                frame, now['boxes'], now['ids'], now['category'], now['scores'])
            out.write(frameDis)

        print('frame {} finished'.format(ID))

    savePath_npy = os.path.join(OUTPUT_DIR, 'output-keypoints-ByteTrack.npy')
    np.save(savePath_npy, results)
    print('Save track results to {}'.format(savePath_npy))
    if args.save_vid:
        out.release()
        print('Save video to {}'.format(savePath))
    cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataPath', type=str, help='Folder which include output.avi and output-keypoints.npy', nargs=1)
    parser.add_argument('--save-vid', action='store_true',
                        help='save video tracking results')
    parser.add_argument("--track_thresh", type=float,
                        default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30,
                        help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float,
                        default=0.8, help="matching threshold for tracking")
    parser.add_argument("--mot20", dest="mot20", default=False,
                        action="store_true", help="test mot20.")
    parser.add_argument('--min_box_area', type=float,
                        default=10, help='filter out tiny boxes')

    args = parser.parse_args()

    Solve(args)
