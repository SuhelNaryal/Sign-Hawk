import cv2
import os
import glob


def clipper(video, words, starts, ends):
    
    video_stream = cv2.VideoCapture(video)
    fps = float(video_stream.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (int(video_stream.get(3)), int(video_stream.get(4)))
    word_idx = 0
    count = 0
    word_count = len(words)

    video_writer = cv2.VideoWriter()

    while True:
        if word_idx >= word_count:
            break

        ret, frame = video_stream.read()
        if not ret:
            break
            
        if count == starts[word_idx]:
            if os.path.exists(words[word_idx]):
                prev_video_count = len(glob.glob(words[word_idx] + '/*.mp4'))
            else:
                os.mkdir(words[word_idx])
                prev_video_count = 0
            output_path = words[word_idx] + '/' + str(prev_video_count) + '.mp4'
            video_writer.open(output_path, fourcc, fps, size)

        if video_writer.isOpened():
            video_writer.write(frame)
        
        if count == ends[word_idx]:
            video_writer.release()
            word_idx += 1
            prev_video_count += 1

        count += 1

    if video_writer.isOpened():
        video_writer.release()


if __name__ == '__main__':
    video = input('video filename: ')
    words = input('enter words list space separated: ').split(' ')
    starts = list(map(int, input('enter start frames list: ').split(' ')))
    ends = list(map(int, input('enter end frames list: ').split(' ')))
    
    clipper(video, words, starts, ends)
    print('clipping done')
