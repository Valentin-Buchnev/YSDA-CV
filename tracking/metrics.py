def iou_score(bbox1, bbox2):
    """Jaccard index or Intersection over Union.

    https://en.wikipedia.org/wiki/Jaccard_index

    bbox: [xmin, ymin, xmax, ymax]
    """

    assert len(bbox1) == 4
    assert len(bbox2) == 4
    
    r1 = bbox1[1]
    c1 = bbox1[0]
    h1 = bbox1[3] - bbox1[1]
    w1 = bbox1[2] - bbox1[0]
    
    r2 = bbox2[1]
    c2 = bbox2[0]
    h2 = bbox2[3] - bbox2[1]
    w2 = bbox2[2] - bbox2[0]
    
    nr = max(0, min(r1 + h1 - 1, r2 + h2 - 1) - max(r1, r2) + 1)
    nc = max(0, min(c1 + w1 - 1, c2 + w2 - 1) - max(c1, c2) + 1)
    
    S = h1 * w1 + h2 * w2 - nr * nc
    
    return nr * nc / S


def motp(obj, hyp, threshold=0.5):
    """Calculate MOTP

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Write code here

        # Step 1: Convert frame detections to dict with IDs as keys
        
        dict_obj = {d[0]: d[1:] for d in frame_obj}
        dict_hyp = {d[0]: d[1:] for d in frame_hyp}

        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        
        for idx_obj, idx_hyp in matches.items():
            if idx_obj in dict_obj and idx_hyp in dict_hyp:
                dist = iou_score(dict_obj[idx_obj], dict_hyp[idx_hyp])
                if dist > threshold:
                    dist_sum += dist
                    match_count += 1
                    del dict_obj[idx_obj]
                    del dict_hyp[idx_hyp]

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        
        iou = []
        for idx_obj, bbox_obj in dict_obj.items():
            for idx_hyp, bbox_hyp in dict_hyp.items():
                metric = iou_score(bbox_obj, bbox_hyp)
                if metric > threshold:
                    iou.append((idx_obj, idx_hyp, metric))

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections

        # Step 5: Update matches with current matched IDs
        
        used_obj = set()
        used_hyp = set()
        for idx_obj, id_hyp, metric in sorted(iou, key=lambda x: x[2], reverse=True):
            if idx_obj not in used_obj and idx_hyp not in used_hyp:
                used_obj.add(idx_obj)
                used_hyp.add(idx_hyp)
                matches[idx_obj] = idx_hyp
                dist_sum += metric
                match_count += 1
                del dict_obj[idx_obj]
                del dict_hyp[idx_hyp]

    # Step 6: Calculate MOTP
    MOTP = dist_sum / match_count

    return MOTP


def motp_mota(obj, hyp, threshold=0.5):
    """Calculate MOTP/MOTA

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0
    missed_count = 0
    false_positive = 0
    mismatch_error = 0

    matches = {}  # matches between object IDs and hypothesis IDs
    
    n = 0

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Step 1: Convert frame detections to dict with IDs as keys
        
        dict_obj = {d[0]: d[1:] for d in frame_obj}
        dict_hyp = {d[0]: d[1:] for d in frame_hyp}
        n += len(dict_obj)

        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        
        for idx_obj, idx_hyp in matches.items():
            if idx_obj in dict_obj and idx_hyp in dict_hyp:
                dist = iou_score(dict_obj[idx_obj], dict_hyp[idx_hyp])
                if dist > threshold:
                    dist_sum += dist
                    match_count += 1
                    del dict_obj[idx_obj]
                    del dict_hyp[idx_hyp]

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        
        iou = []
        for idx_obj, bbox_obj in dict_obj.items():
            for idx_hyp, bbox_hyp in dict_hyp.items():
                metric = iou_score(bbox_obj, bbox_hyp)
                if metric > threshold:
                    iou.append((idx_obj, idx_hyp, metric))
        
        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections     

        # Step 5: If matched IDs contradict previous matched IDs - increase mismatch error

        # Step 6: Update matches with current matched IDs
        
        used_obj = set()
        used_hyp = set()
        for idx_obj, idx_hyp, metric in iou:
            if idx_obj not in used_obj and idx_hyp not in used_hyp:
                used_obj.add(idx_obj)
                used_hyp.add(idx_hyp)
                if idx_obj in matches and matches[idx_obj] != idx_hyp:
                    mismatch_error += 1
                matches[idx_obj] = idx_hyp
                dist_sum += metric
                match_count += 1
                del dict_obj[idx_obj]
                del dict_hyp[idx_hyp]

        # Step 7: Errors
        # All remaining hypotheses are considered false positives
        # All remaining objects are considered misses
        
        false_positive += len(dict_hyp)
        missed_count += len(dict_obj)

    # Step 8: Calculate MOTP and MOTA
    MOTP = dist_sum / match_count
    MOTA = 1 - (missed_count + false_positive + mismatch_error) / n

    return MOTP, MOTA
