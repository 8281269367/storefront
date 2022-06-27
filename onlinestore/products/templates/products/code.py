import cv2
import subprocess
import time
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject, GLib

import threading
import logging


class SensorFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, fps, img_shape, cols, verbosity=1, cap=None, speed_preset='medium', properties={}):
        super(SensorFactory, self).__init__(**properties)
        self.height = int(img_shape[0])
        self.width = int(img_shape[1] * cols)
        self.number_frames = 0
        self.stream_timestamp = 0.0
        self.timestamp = time.time()
        self.dt = 0.0
        self.streamed_frames = 0
        self.verbosity = verbosity
        fps = int(fps)
        self.cap = cap
        self.appsrc = None
        self.duration = 1.0 / fps * Gst.SECOND  # duration of a frame in nanoseconds
        key_int_max = ' key-int-max={} '.format(fps)
        caps_str = 'caps=video/x-raw,format=BGR,width={},height={},framerate={}/1 '.format(self.width,
                                                                                           self.height,
                                                                               fps)
        self.launch_string = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' + caps_str + \
                             ' ! videoconvert' \
                             ' ! video/x-raw,format=I420' \
                             ' ! x264enc' \
                             ' ! rtph264pay config-interval=1 pt=96 name=pay0' \
                             ''
        if self.verbosity > 0:
            print('[INFO]: launch string:\n\t' + self.launch_string.replace(' ! ', '\n\t ! '))

    def set_cap(self, cap):
        self.cap = cap

    def on_need_data(self, src, length):
        # this method executes when client requests data
        if self.cap.isOpened():
            frame = self.cap.get_canvas()
            # print(frame.shape)
            ret = True
            if ret:
                if frame.shape[:2] != (self.height, self.width):
                    frame = cv2.resize(frame, (self.width, self.height))
                data = frame.tostring()
                buf = Gst.Buffer.new_allocate(None, len(data), None)
                buf.fill(0, data)
                buf.duration = self.duration
                buf.pts = buf.dts = int(self.stream_timestamp)
                buf.offset = self.stream_timestamp
                self.number_frames += 1
                self.stream_timestamp += self.duration
                retval = self.appsrc.emit('push-buffer', buf)
                if self.verbosity > 1:
                    next_timestamp = time.time()
                    self.dt += (next_timestamp - self.timestamp)
                    self.streamed_frames += 1
                    self.timestamp = next_timestamp
                    if self.dt > 1:
                        print('[INFO]: Frame {}, FPS = {}'.format(self.number_frames, self.streamed_frames / self.dt))
                        self.streamed_frames = 0
                        self.dt = 0.0
                if retval != Gst.FlowReturn.OK and self.verbosity > 0:
                    print("[INFO]: retval not OK: {}".format(retval))
            elif self.verbosity > 0:
                print("[INFO]: Unable to read frame from cap.")
                # time.sleep(0.05)

    def do_create_element(self, url):
        if self.verbosity > 1:
            request_uri = url.get_request_uri()
            print('[INFO]: stream request on {}'.format(request_uri))
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        self.number_frames = 0
        self.appsrc = rtsp_media.get_element().get_child_by_name('source')
        self.appsrc.connect('need-data', self.on_need_data)  # executes when client requests data


class GstServer2(GstRtspServer.RTSPServer):
    def __init__(self, fps, suffix='test', rtp_port=8554, ip='12.0.0.0', caps=(None,), Sizes=[[300, 300]],
                 speed_preset='medium', verbosity=1, Indexes=[]):
        GObject.threads_init()
        Gst.init(None)
        super(GstServer2, self).__init__(**{})
        self.verbosity = verbosity
        self.rtp_port = "{}".format(rtp_port)
        if int(self.rtp_port) < 1024 and self.verbosity > 0:
            print('[INFO]: Note, admin privileges are required because port number < 1024.')
        self.set_service(self.rtp_port)
        self.speed_preset = speed_preset
        self.caps = caps
        self.factory = [None] * len(self.caps)
        self.suffix = suffix
        self.fps = fps
        self.Sizes = Sizes
        self.Indexes = Indexes
        self.attach(None)
        self.ip = self.get_ip()

        if len(self.suffix):
            self.full_suffix = '/' + self.suffix.lstrip('/')
        else:
            self.full_suffix = ''

        self.connect("client-connected", self.client_connected)

        print('[INFO]: streaming on:\n\trtsp://{}:{}/{}#'.format(self.ip, self.rtp_port, self.suffix))
        # self.send_data = self.factory.send_data
        self.GObject = GObject
        self.Gst = Gst

        self.loop = GObject.MainLoop()

    def set_caps(self, caps):
        if not isinstance(caps, (list, tuple)):
            caps = [caps]
        self.caps = caps

    def create_media_factories(self):
        mount_points = self.get_mount_points()
        All = []
        for i, cap in enumerate(self.caps):
            # mount_points.remove_factory(self.full_suffix + str(i))
            img_shape = self.Sizes[i]
            if len(self.Indexes) == 0:
                N_Index = str(i + 1)
            else:
                N_Index = str(self.Indexes[i])
            factory = SensorFactory(fps=self.fps, img_shape=img_shape, speed_preset=self.speed_preset,
                                    cols=1, verbosity=self.verbosity, cap=cap)
            factory.set_shared(True)
            logging.info('Stream on ' + self.full_suffix + N_Index)
            print('Stream on ' + self.full_suffix + N_Index)

            mount_points.add_factory(self.full_suffix + N_Index, factory)
            self.factory[i] = factory
            All.append(self.full_suffix + str(i))
            self.All = All

    def client_connected(self, gst_server_obj, rtsp_client_obj):
        self.create_media_factories()
        if self.verbosity > 0:
            print('[INFO]: Client has connected')

    def run(self):
        rtsp_server_thread = threading.Thread(target=self.loop.run)
        rtsp_server_thread.daemon = True
        rtsp_server_thread.start()

        # self.loop.run()

    def stop(self):
        self.loop.quit()

    @staticmethod
    def get_ip():
        return subprocess.check_output("hostname -I", shell=True).decode('utf-8').split(' ')[0]


import numpy as np


class Cap_Gstremer:
    def __init__(self):
        self.k = 0
        image = np.zeros((300,300, 3)).astype('uint8')
        self.Last = image.copy()
        A = threading.Thread(target=self.image_create)
        A.daemon = True
        A.start()

    def image_create(self):
        while True:
            image = np.zeros((300,300, 3)).astype('uint8')
            cv2.putText(image, str(self.k), (50,250),cv2.FONT_HERSHEY_SIMPLEX, min(300,300)/(50/1), (0, 0, 255), 10)
            self.k = self.k + 1
            self.Last = image.copy()
            time.sleep(1)

    def isOpened(self):
        return True

    def get_canvas(self):
        A = self.Last
        return A




#######################################################


PRINT_OUT = 0
# cython: language_level=3, boundscheck=False
import multiprocessing as mp
from enum import Enum
import time
import numpy as np
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)

import warnings

warnings.filterwarnings("ignore")

'''Konwn issues
* if format changes at run time system hangs
'''


class StreamMode(Enum):
    INIT_STREAM = 1
    SETUP_STREAM = 1
    READ_STREAM = 2


class StreamCommands(Enum):
    FRAME = 1
    ERROR = 2
    HEARTBEAT = 3
    RESOLUTION = 4
    STOP = 5


class StreamCapture(mp.Process):

    def __init__(self, link, stop, outQueue, framerate, Thermal=0, ID=0, TRES=640):
        """
        Initialize the stream capturing process
        link - rstp link of stream
        stop - to send commands to this process
        outPipe - this process can send commands outside
        """

        super().__init__()
        self.ID = ID
        print("rtsp location is assigned:"+link)
        self.streamLink = link
        self.stop = stop
        self.outQueue = outQueue
        self.Last_Frame = None
        self.Last_Time = time.time()

        self.framerate = framerate
        self.currentState = StreamMode.INIT_STREAM
        self.pipeline = None
        self.source = None
        self.decode = None
        self.convert = None
        self.sink = None
        self.image_arr = None
        self.newImage = False
        self.frame1 = None
        self.frame2 = None
        self.Thermal = Thermal
        self.TRES = TRES
        self.err = 0

    def gst_to_opencv(self, sample):
        buf = sample.get_buffer()
        caps = sample.get_caps()

        # Print Height, Width and Format
        # print(caps.get_structure(0).get_value('format'))
        # print(caps.get_structure(0).get_value('height'))
        # print(caps.get_structure(0).get_value('width'))

        arr = np.ndarray(
            (caps.get_structure(0).get_value('height'),
             caps.get_structure(0).get_value('width'),
             3),
            buffer=buf.extract_dup(0, buf.get_size()),
            dtype=np.uint8)
        return arr

    def new_buffer(self, sink, _):
        sample = sink.emit("pull-sample")
        arr = self.gst_to_opencv(sample)
        self.image_arr = arr
        self.newImage = True
        return Gst.FlowReturn.OK

    def run(self):
        # Create the empty pipeline
        self.pipeline = Gst.parse_launch(
            'rtspsrc name=m_rtspsrc ! rtph264depay name=m_rtph264depay ! avdec_h264 name=m_avdech264 ! videoscale method=0 ! video/x-raw,width=300,height=300 ! videoconvert name=m_videoconvert ! videorate name=m_videorate ! appsink name=m_appsink')

        # source params
        print('gst pipeline started for decode thread')
        self.source = self.pipeline.get_by_name('m_rtspsrc')
        self.source.set_property('latency', 0)
        self.source.set_property('location', self.streamLink)
        self.source.set_property('protocols', 'tcp')
        self.source.set_property('retry', 100)
        self.source.set_property('timeout', 100)
        self.source.set_property('tcp-timeout', 5000000)
        self.source.set_property('drop-on-latency', 'true')

        # decode params
        self.decode = self.pipeline.get_by_name('m_avdech264')
        self.decode.set_property('max-threads', 2)
        self.decode.set_property('output-corrupt', 'false')

        # convert params
        self.convert = self.pipeline.get_by_name('m_videoconvert')

        # framerate parameters
        self.framerate_ctr = self.pipeline.get_by_name('m_videorate')
        self.framerate_ctr.set_property('max-rate', self.framerate / 1)
        self.framerate_ctr.set_property('drop-only', 'true')

        # sink params
        self.sink = self.pipeline.get_by_name('m_appsink')

        # Maximum number of nanoseconds that a buffer can be late before it is dropped (-1 unlimited)
        # flags: readable, writable
        # Integer64. Range: -1 - 9223372036854775807 Default: -1
        self.sink.set_property('max-lateness', 500000000)

        # The maximum number of buffers to queue internally (0 = unlimited)
        # flags: readable, writable
        # Unsigned Integer. Range: 0 - 4294967295 Default: 0
        self.sink.set_property('max-buffers', 5)

        # Drop old buffers when the buffer queue is filled
        # flags: readable, writable
        # Boolean. Default: false
        self.sink.set_property('drop', 'true')

        # Emit new-preroll and new-sample signals
        # flags: readable, writable
        # Boolean. Default: false
        self.sink.set_property('emit-signals', True)

        # # sink.set_property('drop', True)
        # # sink.set_property('sync', False)

        # The allowed caps for the sink pad
        # flags: readable, writable
        # Caps (NULL)
        caps = Gst.caps_from_string(
            'video/x-raw, format=(string){BGR, GRAY8}; video/x-bayer,format=(string){rggb,bggr,grbg,gbrg}')
        self.sink.set_property('caps', caps)

        if not self.source or not self.sink or not self.pipeline or not self.decode or not self.convert:
            if PRINT_OUT == 1:
                print("Not all elements could be created.")
            self.stop.set()

        self.sink.connect("new-sample", self.new_buffer, self.sink)

        # Start playing
        self.pipeline.set_state(Gst.State.NULL)
        time.sleep(0.2)
        self.pipeline.set_state(Gst.State.READY)
        time.sleep(0.2)
        self.pipeline.set_state(Gst.State.PAUSED)
        ret = self.pipeline.set_state(Gst.State.PLAYING)

        time.time()
        qqq = 0
        while True:
            status, state, pending = self.pipeline.get_state(0)
            if PRINT_OUT == 1:
                print(qqq)
            if state == Gst.State.PAUSED:
                time.sleep(1)
                qqq = qqq + 1
                # print(state)
            else:
                break
            if qqq > 20:
                self.stop.set()
                break

        if ret == Gst.StateChangeReturn.FAILURE:
            if PRINT_OUT == 1:
                print("Unable to set the pipeline to the playing state.")
            self.stop.set()

        # Wait until error or EOS
        bus = self.pipeline.get_bus()
        Tries = 5
        while True:

            if self.stop.is_set():
                if PRINT_OUT == 1:
                    print('Stopping CAM Stream by main process')
                break

            message = bus.timed_pop_filtered(10000, Gst.MessageType.ANY)
            # print "image_arr: ", image_arr
            time.sleep(0.02)
            if self.image_arr is not None and self.newImage is True:
                self.Last_Frame = self.image_arr
                self.Last_Time = time.time()

                if not self.outQueue.full():
                    # print("\r adding to queue of size{}".format(self.outQueue.qsize()), end='\r')
                    self.outQueue.put((StreamCommands.FRAME, self.image_arr), block=False)

                self.image_arr = None

            if message:
                if message.type == Gst.MessageType.ERROR:
                    err, debug = message.parse_error()
                    if PRINT_OUT == 1:
                        print("Error received from element %s: %s" % (
                            message.src.get_name(), err))
                        print("Debugging information: %s" % debug)
                        print("End-Of-Stream reached.")

                    self.err = 1
                    break
                elif message.type == Gst.MessageType.EOS:
                    if PRINT_OUT == 1:
                        print("End-Of-Stream reached.")

                    self.err = 1
                    break
                elif message.type == Gst.MessageType.STATE_CHANGED:
                    if isinstance(message.src, Gst.Pipeline):
                        old_state, new_state, pending_state = message.parse_state_changed()
                        if PRINT_OUT == 1:
                            print("Pipeline state changed from %s to %s." %
                                  (old_state.value_nick, new_state.value_nick))
                else:
                    1 + 1
                    # print("Unexpected message received.")

        if PRINT_OUT == 1:
            print('terminating cam pipe')
        self.stop.set()
        self.pipeline.set_state(Gst.State.PAUSED)
        time.sleep(0.1)
        self.pipeline.set_state(Gst.State.READY)
        time.sleep(0.1)

        self.pipeline.set_state(Gst.State.NULL)
        # time.sleep(0.1)
        #
        # self.pipeline.ref_count
        # self.source.ref_count
        # self.source.unref()
        self.pipeline.unref()
        # self.pipeline.remove(self.source)

        time.sleep(0.5)
        if PRINT_OUT == 1:
            print('Quit')


import threading


class VidStreamClass_Test:
    def __init__(self, Address, q1):
        self.Cam_idx = q1
        self.Last_Frame = np.zeros((300, 300, 3))
        # Current Cam
        self.camProcess = None
        self.cam_queue = None
        self.stopbit = None
        self.camlink = Address
        self.framerate = 25
        self.cam_queue = mp.Queue(maxsize=2)
        self.stopbit = mp.Event()
        self.terminate = 0
        self.Thermal = 1

        # self.startMain()
        self.thread = threading.Thread(target=self.startMain)
        self.thread.daemon = True
        self.thread.start()
        print("Thread startMain started")

        time.sleep(3)

        self.get_frame()
        self.thread1 = threading.Thread(target=self.Loop_get_frame)
        self.thread1.daemon = True
        self.thread1.start()
        print("Thread Loop_get_frame started")

    def startMain(self):
        if PRINT_OUT == 1:
            print('Started Thread')
        self.camProcess = StreamCapture(self.camlink,
                                        self.stopbit,
                                        self.cam_queue,
                                        self.framerate, self.Thermal, ID=self.Cam_idx)  # ,TRES=self.TRES)
        self.camProcess.err = 0
        self.camProcess.daemon = True
        self.camProcess.start()
        while True:
            time.sleep(0.3)
            if self.terminate == 1:
                self.stopbit.set()

                time.sleep(0.1)
                self.camProcess.terminate()

                break
        if PRINT_OUT == 1:
            print('Finished start Main')

    def get_frame(self):
        cmd = 0
        val = self.Last_Frame
        if not self.cam_queue.empty():
            try:
                cmd, val = self.cam_queue.get(block=False)
            except:
                time.sleep(0.05)
                try:
                    cmd, val = self.cam_queue.get(block=False)
                except:

                    1 + 1

            self.Last_Frame = val
            self.Time_of_Last_Frame = time.ctime()
            self.Computer_Time_of_Last_Frame = time.time()
            self.Camera_Stuck = 0

        return cmd, val

    def isOpened(self):
        return True

    def Loop_get_frame(self):
        while True:
            try:
                self.get_frame()
            except:
                1 + 1
            time.sleep(0.01)
            time.sleep(1 / self.framerate)




################################################################



import Encoder as enc
import Decoder as dec
import cv2

Cam1 = enc.Cap_Gstremer()

kk = 1
speed_preset = 'fast'
rtsp_server = enc.GstServer2(fps=25, Sizes=[[300, 300]], speed_preset=speed_preset, caps=[Cam1], suffix='video',
                             verbosity=0, rtp_port=554, ip='10.5.1.130')
rtsp_server2 = enc.GstServer2(fps=25, Sizes=[[300, 300]], speed_preset=speed_preset, caps=[Cam1], suffix='video',
                              verbosity=0, rtp_port=555, ip='10.5.1.130')
# rtsp_server3 = enc.GstServer2(fps=25, Sizes=[[300, 300]], speed_preset=speed_preset, caps=[Cam1], suffix='video',
#                               verbosity=0, rtp_port=556, ip='10.5.1.130')
# rtsp_server4 = enc.GstServer2(fps=25, Sizes=[[300, 300]], speed_preset=speed_preset, caps=[Cam1], suffix='video',
#                               verbosity=0, rtp_port=557, ip='10.5.1.130')
# rtsp_server5 = enc.GstServer2(fps=25, Sizes=[[300, 300]], speed_preset=speed_preset, caps=[Cam1], suffix='video',
#                               verbosity=0, rtp_port=558, ip='10.5.1.130')
# rtsp_server6 = enc.GstServer2(fps=25, Sizes=[[300, 300]], speed_preset=speed_preset, caps=[Cam1], suffix='video',
#                               verbosity=0, rtp_port=559, ip='10.5.1.130')

X = dec.VidStreamClass_Test('rtsp://192.168.8.137:554/video1', 1)
X1 = dec.VidStreamClass_Test('rtsp://192.168.8.137:555/video1', 1)
print(X.camlink)
print(X1.camlink)
# X2 = dec.VidStreamClass_Test('rtsp://192.168.8.137:556/video1', 1)
# X3 = dec.VidStreamClass_Test('rtsp://192.168.8.137:557/video1', 1)
# X4 = dec.VidStreamClass_Test('rtsp://192.168.8.137:558/video1', 1)
# X5 = dec.VidStreamClass_Test('rtsp://192.168.8.137:559/video1', 1)
p = 0
while True:
    image = Cam1.Last.copy()
    # image2 = Cam2.Last.copy()
    # image3 = Cam3.Last.copy()
    # image4 = Cam4.Last.copy()
    # image5 = Cam5.Last.copy()
    # image6 = Cam6.Last.copy()
    cv2.imshow('RTSP video Encoded', image)
    # cv2.imshow('RTSP video Encoded 2', image2)
    # cv2.imshow('RTSP video Encoded 3', image3)
    # cv2.imshow('RTSP video Encoded 4', image4)
    # cv2.imshow('RTSP video Encoded 5', image5)
    # cv2.imshow('RTSP video Encoded 6', image6)

    IM0 = X.Last_Frame + 0
    cv2.putText(IM0, str(p), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.imshow('Decode0', IM0)

    IM1 = X1.Last_Frame + 0
    cv2.putText(IM1, str(p), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.imshow('Decode1', IM1)

    # IM2 = X2.Last_Frame + 0
    # cv2.putText(IM2, str(p), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    # cv2.imshow('Decode2', IM2)

    # IM3 = X3.Last_Frame + 0
    # cv2.putText(IM3, str(p), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    # cv2.imshow('Decode3', IM3)
    #
    # IM4 = X4.Last_Frame + 0
    # cv2.putText(IM4, str(p), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    # cv2.imshow('Decode4', IM4)
    #
    # IM5 = X5.Last_Frame + 0
    # cv2.putText(IM5, str(p), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    # cv2.imshow('Decode5', IM5)

    p = p + 1

    k = cv2.waitKey(1)
    if k == 27:
        break
cv2.destroyAllWindows()
