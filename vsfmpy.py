"""
A python wrapper by Chris Buurma for for Visual Structure from Motion:
http://ccwu.me/vsfm/

Visual Structure From Motion by
Changchang Wu
ccwu@cs.washington.edu

This code serves as a command-line socket-interface wrapper for VSFM.
Commands are sent based on their location in the windows GUI, and a set of socket commands related to that GUI menu
are available in the VSFM docs.

The vsfm_data file is from a previous python 2.7 module

This simple module also contains some handy functions for dealing with VSFM binary 'sift' files, and supports intergration
with OpenCV's keypoints class
"""

import struct
import time
import subprocess
import socket
import os
import logging
name = __name__
loglevel = logging.INFO   # Adust the logging level here.
logger = logging.getLogger(name)
logger.setLevel(loglevel)
console_handler = logging.StreamHandler()
console_handler.setLevel(loglevel)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

try:
    from vsfm_data import vsfm_command_dict, default_path
except ImportError:
    try:
        from .vsfm_data import vsfm_command_dict, default_path
    except ImportError:
        logger.error("Could not import from vsfm_data")
try:
    from cv2 import KeyPoint
    from numpy import ndarray, pi
    tau = 2 * pi
    opencv=True
except:
    logger.error("Could not import openCV and numpy.")
    tau = 2 * 3.141592653589793
    class KeyPoint(object):
        """
        This class is defined only if openCV is missing. These objects will serve as placeholders for the keypoint class
        that is normally used by openCV.
        """
        def __init__(self, x, y, size, angle=-1, response=0, octave=0, class_id=-1):
            """
            Setup some attributes for the class
            :param x: x pixel position of the keypoint
            :param y: y pixel position of the keypoint
            :param size: relative size in pixels of the keypoint
            :param angle: its relative angle
            :param response: carry over from openCV
            :param octave: On a pyramid's scale how is it with respect to the rest
            :param class_id: carry over from openCV
            """
            self.x = x
            self.y = y
            self.pt = (x, y)
            self.size = size
            self.angle = angle
            self.response = response
            self.octave = octave
            self.class_id = class_id
    opencv = False

def read_vsfm_sift(filename):
    """
    This function reads in a vsfm-generated '.sift' file and returns a keypoint list and their descriptions
    :param filename: a vsfm generated .sift file
    :return: tuple containing a keypoint list, and descriptions for those keypoints
    """
    logger.info("Now reading vsfm sift binary file " + filename)
    with open(filename, 'rb') as fileobj:
        header_format = 'c'*8 + 'L'*3 # 'method, version, # features, the number 5 and the number 128. No idea why.
        loc_format = 'ffBBBff'  # x, y, color rgb, scale, orientation in radians
        desc_format = 'B'*128  # description vector as chars for some reason
        header_bin = fileobj.read(struct.calcsize('s'*8 + 'L'*3))
        header= struct.unpack_from(header_format, header_bin)
        sift_type = ''.join([ch.decode('utf-8') for ch in header[0:8]])
        logger.debug("Sift type is: " + sift_type)
        nfeatures = header[8]
        logger.debug("Detecting " + str(nfeatures) + " features in this file.")
        keypoints = []
        descriptions = []
        for i in range(nfeatures):
            loc_bin = fileobj.read(4*5)
            location = struct.unpack_from(loc_format, loc_bin)
            keypoints.append(location)
        logger.debug(str(len(keypoints)) + " keypoints found")

        for _ in range(nfeatures):
            desc_bin = fileobj.read(struct.calcsize(desc_format))
            desc = struct.unpack_from(desc_format, desc_bin)
            descriptions.append(desc)
        logger.debug(str(len(descriptions)) + " descriptions found, two examples:")
        buffer = fileobj.read()
        logger.debug("The rest of the buffer is:" + str(buffer))
        kp_list = [KeyPoint(kp[0], kp[1], kp[5], kp[6]*360/tau) for kp in keypoints]
        return kp_list, descriptions

def write_vsfm_sift(keypoints, descriptors=None, filename=None):
    """
    This function writes the keypoints and descriptors to a VSFM-readable '.sift' file.
    Note that while it writes .sift files, there's no need that the SIFT algorithm be used to generate keypoints
    :param keypoints: List of Keypoint objects
    :param descriptors: List of vectors which are each descriptors of those keypoints. Used for matching
    :param filename: Filename to save this file to. If not provided, will be 'features.sift' in the same path
    :return:
    """
    if filename is None:
        filename = 'features.sift'
    with open(filename, 'wb') as fileobj:
        logger.debug("Now writing vsfm binary 'sift' file for features")
        # format strings for the binary daata
        header_format = 'c'*8+'L'*3
        loc_format = 'ffBBBff'
        desc_format = 'B'*128
        #Header
        header_bin = struct.pack(header_format, *[ch.encode('utf-8') for ch in 'SIFTV4.0'], len(keypoints), 5, 128)
        fileobj.write(header_bin)
        # Now for features
        for kp in keypoints:
            loc_bin = struct.pack(loc_format, kp.pt[0], kp.pt[1], *[0, 0, 0], kp.size, kp.angle*tau/360)
            fileobj.write(loc_bin)
        for desc in descriptors:
            if type(desc) is list or (opencv and type(desc) is ndarray):
                desc = [int(val) for val in desc]
            if len(desc) < 128:
                desc += [0]*(128-len(desc))
            elif len(desc) > 128:
                desc = desc[0:128]
            desc_bin = struct.pack(desc_format, *desc)
            fileobj.write(desc_bin)
        fileobj.write(b'\xFFEOF')
    return True

def write_feature_matches(matches_list, filenames, match_path=None):
    """
    This function writes the list of matched features to a keypoint matching text file which is readable bt VSFM
    :param matches_list: List of keypoint matches
    :param filenames: filenames of the .sift files which are matched.
    :param match_path: Output path of the text file to be read by VSFM
    :return: None
    """
    if match_path is None:
        match_path = 'kp_matches.txt'
    with open(match_path, 'w') as fileobj:
        for i, j, matches in matches_list:
            fileobj.write(filenames[i] + " " + filenames[j] + " " + str(len(matches)) + "\n")
            fileobj.write(" ".join([str(match.queryIdx) for match in matches]) + "\n")
            fileobj.write(" ".join([str(match.trainIdx) for match in matches]) + "\n")

def start_vsfm(port=None, vsfm_binary_path=default_path):
    """
    Starts VSFM, binds it to a socket, opens the socket interface, sets up a logger and waits.
    :param port: Port number to open, defaults to a random one
    :param vsfm_binary_path: the path to VSFM.exe, defaults from the vsfm_data file
    :return: port that was opened
    """
    # 'start program'
    if port is None:
        tmp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tmp_sock.bind(('',0))
        port = int(tmp_sock.getsockname()[1])
        tmp_sock.close()
        logger.info("Binding to port " + str(port))
    cmd = '"{}" listen+log {}'.format(vsfm_binary_path, port)
    # Opens up VSFM and begins the socket.
    logger.debug("Sending cmd: " + cmd)
    vsfm_subprocess = subprocess.Popen(cmd, shell=True) # this needs changed from shell=True
    return port

def open_socket(port, host='localhost', wait=True):
    """
    Opens the socket over the host, and specifies if we have to wait for the connection
    :param port: Port to be opened
    :param host: machine that is hosting vsfm. Defaults to localhost
    :param wait: True/false if we need to wait to connect
    :return:
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    for _ in range(10):
        try:
            sock.connect((host, port))
            logger.debug("Socket connected on " + host + " to port " + port)
            break
        except:
            time.sleep(0.1)
    time.sleep(0.05)
    return sock

def send_vsfm_command_num(open_socket, number, param=None, wait=False, timeout=60):
    """
    This function sends a VSFM command via its direct number from the documentation of VSFM.
    You shouldn't need this mfunction, but it is called from ::send_vsfm_command_tup
    :param open_socket: The socket that is already opened
    :param number: the integer comamand you want to send
    :param param: any single parameter to send with it
    :param wait: if we should wait for a response
    :param timeout: how long we should wait for a response at maximum
    :return:
    """
    logger.debug("Sending command #" + str(number))
    if param is None:
        cmd = str(number) + '\n'
    else:
        cmd = str(number) + " " + param + '\n'
    cmd_bytes = cmd.encode()
    open_socket.sendall(cmd_bytes)
    if wait:
        wait_until_complete(open_socket, timeout)
    return True

def send_vsfm_command_tup(open_socket, tuple_command, param=None, wait=False, timeout=60):
    """
    Sends a VSFM command as a tuple object. This tuple is just strings corresponding to the menu invoked.
    :param open_socket: The open socket to VSFM
    :param tuple_command: The command to be sent, e.g ('file', 'open_multi_images')
    :param param: The parameter if any sent along the command line
    :param wait: weather or not to wait for the command to finish
    :param timeout: total timeout to wait for a command to finish
    :return:
    """
    logger.debug("Sending command: " + str(tuple_command))
    try:
        last = vsfm_command_dict
        for command in tuple_command:
            last = last[command]
        send_vsfm_command_num(open_socket, last, param, wait, timeout)
    except ValueError:
        logger.error("Command '" + str(tuple_command) + "' not found")

def wait_until_complete(sock, timeout=None):
    """
    Waits until VSFM receives a complete flag
    :param timeout: Time in seconds that all processes will be completed by. Suggests very long time, like 10min
    :return:
    """
    # time.sleep(0.2)
    # set socket non-blocking
    sock.setblocking(0)
    time.sleep(0.1)
    waiting = True
    begin = time.time()
    vsfm_complete_flags = ['*command processed*', 'done', 'finished']
    bytes_rec = b''
    while waiting:
        # Read the buffer until its empty
        # Then check the last set of chars for one of the complete flags
        try:
            bytes_rec = sock.recv(256)
            # If the socket has no data, this is an error
            logger.debug("Buffer is: " + str(bytes_rec))
            time.sleep(0.05)
        except:
            logger.debug("Buffer is empty, checking for complete flags.")
            time.sleep(0.2)
            for flag in vsfm_complete_flags:
                if str(bytes_rec).find(flag) == -1:
                    time.sleep(0.2)
                # exit command not yet processed
                else:
                    logger.info("VSFM close flag found: " + str(bytes_rec))
                    waiting = False
        if timeout is not None:
            if time.time() - begin > timeout:
                logger.error("Timeout exceeded")
                waiting = False

def vsfm_of_img_dir(images_path = r'../testing/image sets/kermit', close=True):
    """
    This function is mostly an example of how to run a complete VSFM on a directory of images.
    :param images_path: The folder directory containing a series of pictures
    :param close: Whether or not we close when we're done
    :return:
    """
    images = [file for file in os.listdir(images_path) if any([file.endswith(ext) for ext in ['.jpg', '.png']])]
    logger.info(str(len(images)) + " images found in path " + images_path)
    port = start_vsfm()
    open_sock = open_socket(port)

    # Begin data processing
    for image in images:
        send_vsfm_command_tup(open_sock, ('file', 'open_multi_images'), images_path + os.sep+ image)
    send_vsfm_command_tup(open_sock, ('view', 'image_thumbnails'))
    send_vsfm_command_tup(open_sock, ('sfm', 'pairwise', 'compute_missing_match'), wait=True)
    send_vsfm_command_tup(open_sock, ('sfm', 'reconstruct_sparse'), wait=True)
    send_vsfm_command_tup(open_sock, ('sfm', 'reconstruct_dense'), images_path + os.sep + os.path.basename(images_path) + "_3D", wait=True)
    send_vsfm_command_tup(open_sock, ('view', 'dense_3d_points'))
    if close:
        send_vsfm_command_tup(open_sock, ('file', 'exit_program'), wait=True)
    open_sock.close()

if __name__ == '__main__':
    from tkinter import filedialog
    image_path = filedialog.askdirectory()
    vsfm_of_img_dir(image_path, close=False)
