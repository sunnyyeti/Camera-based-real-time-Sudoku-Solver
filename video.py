# -*- coding: utf-8 -*-
import cv2
from imageProcess import *
from Classfier import *
from Sudoku import solve
show_border = True
show_grids = True
show_solution = True
save = True


def video_test():
    mapped_size = (252, 252)
    global_step = 20000
    pred_func = get_trained_model("gray_model_baseline", global_step)
    cap = cv2.VideoCapture(0)

    valid_cnt = 0
    invalid_cnt = 0
    hist_digits_flag = []
    block_to_centre = {}
    cached_digits_flag = None
    cached_solution = None
    reset = True
    while (cap.isOpened()):
        ret, frame = cap.read()
        recs = get_recs(frame)

        if len(recs)>0:
            valid=False
            valid_rec = None
            for rec in recs:
                digits_flag = []
                rot_mat = get_rot_matrix(rec[0], mapped_size, reverse=False)
                mapped = cv2.warpPerspective(frame, rot_mat, mapped_size)
                binary_mapped = preprocess_sudoku_grid(mapped)
                binary_blocks = split_2_blocks(binary_mapped,9,9)
                for i, b in enumerate(binary_blocks):
                    cur_block = b.reshape(28,28)
                    flag,centre = catch_digit_center(cur_block,(16,20))
                    digits_flag.append(flag)
                    if flag:
                        block_to_centre.setdefault(i,[]).append(centre.flatten())
                if sum(digits_flag) >= 17: #if valid digits in the sudoku is less than 17 then we think it as invalid
                    valid = True
                    invalid_cnt = 0
                    valid_rec = rec
                    if not reset and np.sum(digits_flag^cached_digits_flag)<5 and cached_solution is not None:
                        block_to_centre.clear()
                        if show_solution:
                            #print "cached!"
                            mapped = write_solution(mapped, merged_digits_flag, cached_solution)

                        if show_grids:
                            mapped = add_v_h_grids(mapped, 9, 9)

                        frame = reflect_to_orig(frame, rot_mat, mapped)
                        if show_border:
                            frame = add_rectangular(frame, valid_rec[0])

                        cv2.imshow('frame',frame)
                    elif not reset:
                        #print "reset 1"
                        reset = True
                        valid_cnt = 0
                        hist_digits_flag = []
                        #block_to_centre.clear()
                        cached_solution = None
                        cached_digits_flag = None
                    break
            if not valid:
                invalid_cnt += 1
                if invalid_cnt >=3:
                    invalid_cnt = 3
                    reset = True
                    #print "reset 2"
                    valid_cnt = 0
                    hist_digits_flag = []
                    block_to_centre.clear()
                    cached_solution = None
                    cached_digits_flag = None
                cv2.imshow('frame', frame)
            elif reset:
                valid_cnt+=1
                hist_digits_flag.append(digits_flag)
                if valid_cnt > 10:
                    #print "detected!"##continuous valid frame is larger than 10, then we think there is a valid sudoku. The reognized digits are determined by the 10 frames to reduce bias.
                    merged_digits_flag = np.sum(np.array(hist_digits_flag),axis=0).astype(np.bool)
                    #print "args true", np.argwhere(merged_digits_flag==True).flatten()
                    #print "dict keys", block_to_centre.keys()
                    cached_digits_flag = merged_digits_flag
                    reset = False
                    pre_centre_blocks = []
                    for i in xrange(9*9):
                        if merged_digits_flag[i]:
                            pre_centre_blocks.append(np.mean(block_to_centre[i],axis=0))
                    pred_labels = pred_func(pre_centre_blocks)+1
                    sudoku = np.zeros(81)
                    sudoku[merged_digits_flag] = pred_labels
                    sudoku = sudoku.astype(np.int)
                    answer = solve(''.join(map(str,sudoku)))
                    #print "answer is ", answer
                    if answer is not False and show_solution:
                        cached_solution = answer
                        mapped = write_solution(mapped, merged_digits_flag, answer)
                    if show_grids:
                        mapped = add_v_h_grids(mapped,9,9)

                    frame = reflect_to_orig(frame, rot_mat, mapped)
                    if show_border:
                        frame = add_rectangular(frame,valid_rec[0])
                    cv2.imshow('frame', frame)
        else:
            invalid_cnt += 1
            if invalid_cnt >= 3:
                invalid_cnt = 3
                reset = True
                #print "reset 3"
                valid_cnt = 0
                hist_digits_flag = []
                block_to_centre.clear()
                cached_solution = None
                cached_digits_flag = None
            cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_test()
