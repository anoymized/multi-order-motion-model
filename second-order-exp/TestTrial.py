"""Measure coherence threshold of motion direction 
using Staircase procedure, dot direction: 0 (LTR)"""

from __future__ import absolute_import, division, print_function
from numpy import matlib, inf
import numpy as np
import math
from psychopy.tools.filetools import fromFile, toFile
from psychopy.hardware import keyboard
import easydict
from builtins import range
import pandas as pd
import gc
import matplotlib
import os
from psychopy import core, visual, gui, data, event, logging


# initialize
def get_config():
    config = easydict.EasyDict({
        "image_size": 1024,
        "number_of_frame": 30,
        "probe_onset_frame": 14,  # 14 to 15
        "frame_rate": 30,
        "upampling_factor": 1.2,  # we enlarge the MPI movie by 2 times
        'aperture': 300,  # radius of aperture
        'an2px': 50,
        "ratM": 1,
        'ctlsize': 300,
        'NumTrials': 36,  # number of trials for each location
        "feedback": True,
        "modulation_num": 9,
        "img_root": ".\\MOVZT\\image\\",
        "repetition_per_trail": 1,

    })
    return config


EnvCfg = get_config()
EnvCfg.upampling_factor = 1.5
matplotlib.use('Qt5Agg')  # change this to control the plotting 'back end'
try:  # try to get a previous parameters file
    expInfo = fromFile('lastParams.pickle')
except:  # if not there then use a default set
    expInfo = {'Observer': '01', 'Movie': '1'}
expInfo = {'Observer': '01', 'Movie': '01'}

expInfo['dateStr'] = data.getDateStr()  # add the current time

# present a dialogue to change params

dlg = gui.DlgFromDict(expInfo, title='MPI HS FlipGrid', fixed=['dateStr'])
if dlg.OK:
    toFile('lastParams.pickle', expInfo)  # save params to file for next time
else:
    core.quit()  # the user hit cancel so exit

MOV = str('%01d' % int(expInfo['Movie']))
# import condition list from excel
file = '.\\MOVZT\\test2\\NewConvention_Final' + '.xlsx'
# file = 'PreselctedGT_movie' + str(MOV) + '_FlipGridHSV' + '.xlsx'
pdSetupFile = pd.read_excel(file)
MovI = pdSetupFile['MovI'].tolist()  # Scene number
GTX = pdSetupFile['picX'].tolist()  # this is GT center, not the presenting center
GTY = pdSetupFile['picY'].tolist()
Angle = pdSetupFile['angle'].tolist()
Radius = pdSetupFile['radius'].tolist()
flowU = pdSetupFile['flowU'].tolist()
flowV = pdSetupFile['flowV'].tolist()
flipType = pdSetupFile['fliptype'].tolist()
R = pdSetupFile['R'].tolist()
G = pdSetupFile['G'].tolist()
B = pdSetupFile['B'].tolist()
Modulation = pdSetupFile['modulation'].tolist()

# making stimuli
stimList = []
for ctr in range(0, EnvCfg.NumTrials):  # TODO 25space*3time*4flip
    stimList.append({'mov': MovI[ctr], 'Mod': Modulation[ctr], 'GTX': GTX[ctr], 'GTY': GTY[ctr],
                     'Angle': Angle[ctr], 'Radius': Radius[ctr], 'flowU': flowU[ctr], 'flowV': flowV[ctr],
                     'FT': flipType[ctr],
                     'R': R[ctr],
                     'G': G[ctr], 'B': B[ctr]})

# organize them with the trial handler
trials = data.TrialHandler(stimList, 1, method='random',
                           extraInfo={'participant': expInfo['Observer'], 'movie': expInfo['Movie']})

win = visual.Window(allowGUI=True, color=(0, 0, 0), allowStencil=True, monitor='testMonitor', winType='pyglet',
                    units='pix', fullscr=True)

refresh_rate = win.getActualFrameRate()
assert refresh_rate > 25, "Please set the monitor refresh rate up to 30 Hz"
print("current win frame rate：{} Hz".format(refresh_rate))
# assert refresh_rate >= 30.0, "Please set the monitor refresh rate up to 30 Hz"
# calculate duration of each frame
frame_dur = 1.0 / refresh_rate
print("current win frame duration：{} s".format(frame_dur))

myMouse = event.Mouse()  # will use win by default
# and some handy clocks to keep track of time
globalClock = core.Clock()
kb = keyboard.Keyboard()
core.rush(True, realtime=True)
visual.useFBO = True  # if available (try without for comparison) Framebuffer
disable_gc = True  # suspend the garbage collection
process_priority = 'realtime'  # 'high' or 'realtime', 'normal'

if process_priority == 'normal':
    pass
elif process_priority == 'high':
    core.rush(True)
elif process_priority == 'realtime':
    # Only makes a diff compared to 'high' on Windows.
    core.rush(True, realtime=True)
if disable_gc:
    gc.disable()

fileName = expInfo['Observer'] + '_' + expInfo['Movie'] + '_' + expInfo['dateStr']
pathPrefix = os.getcwd()
MessageIni = visual.TextStim(win, pos=[0, 0],
                             text='Initializing, please wait...',
                             color=(-1.0, -1.0, -1.0), autoDraw='False')

MessageIni.draw()
win.flip()
MessageIni.autoDraw = False
message1 = visual.TextStim(win, pos=[0, 300],
                           text='Please adjust rectangle size to match credit card size by '
                                'using four "arrow" keys, press "space" to continue',
                           color=(-1.0, -1.0, -1.0))
message2 = visual.TextStim(win, pos=[0, 250], text='cmperpixel ', color=(-1.0, -1.0, -1.0))
message3 = visual.TextStim(win, pos=[0, 200], text='viewingdistance ', color=(-1.0, -1.0, -1.0))
message4 = visual.TextStim(win, pos=[0, 150], text='press "space" to continue', color=(-1.0, -1.0, -1.0))

message1.autoDraw = 'True'
message2.autoDraw = 'True'
message3.autoDraw = 'True'
message4.autoDraw = 'False'

# Setting for invisible control panel
CtlMaxSp = np.log2(20 + 1)  # 21 pixels/frame, log scale GT is 1~20 pixels/frame   using 21 since 2 ^0=1, so we minus 1
# to control speed
Ctlsize = EnvCfg.ctlsize * EnvCfg.ratM  # radius
ConC = np.array([.0, .0])

# Setting for pink noise
noisesize = [round(120 * EnvCfg.ratM), round(120 * EnvCfg.ratM)]  # pixels ~=3VA
# this is from spatialPattern.m by Jon Yearsley to generate pink noise
Beta = -2
x = np.arange(0, int(math.floor(noisesize[0]) / 2 + 1))
y = -(np.arange(int(math.ceil(noisesize[0] / 2) - 1), 0, -1))
u = np.concatenate((x, y)) / noisesize[0]
u = np.matlib.repmat(u, noisesize[1], 1)
u = u.T
x = np.arange(0, int(math.floor(noisesize[1]) / 2 + 1))
y = -(np.arange(int(math.ceil(noisesize[1] / 2) - 1), 0, -1))
v = np.concatenate((x, y)) / noisesize[1]
v = np.matlib.repmat(v, noisesize[0], 1)
S_f = (u ** 2 + v ** 2) ** (Beta / 2)
S_f[S_f == inf] = 0
phi = np.random.rand(noisesize[0], noisesize[1])
x = np.fft.ifft2(S_f ** 0.5 * (np.cos(2 * math.pi * phi) + ((np.sin(2 * math.pi * phi)) * 1j)));
x = x.real
ranx = np.max(x) - np.min(x)
x = (((x - np.min(x)) / ranx) * 2) - 1
noiseTexture = x
# noiseTexture = np.random.rand(noisesize, noisesize) * 2.0 - 1
noise = visual.GratingStim(win, tex=noiseTexture, mask='circle',
                           size=(noisesize[0], noisesize[1]), pos=(0, 0),
                           interpolate=False, autoLog=False, opacity=0.02)

# this is place holder with four dots/single dot
probelo = np.floor(noisesize[0] / 2) - 1
Probe1 = visual.GratingStim(win, tex="none", mask="circle", pos=(probelo, 0), size=(5, 5), color=(1, -1, -1),
                            autoDraw=True)
Probe2 = visual.GratingStim(win, tex="none", mask="circle", pos=(0, probelo), size=(5, 5), color=(1, -1, -1),
                            autoDraw=True)
Probe3 = visual.GratingStim(win, tex="none", mask="circle", pos=(-probelo, 0), size=(5, 5), color=(1, -1, -1),
                            autoDraw=True)
Probe4 = visual.GratingStim(win, tex="none", mask="circle", pos=(0, -probelo), size=(5, 5), color=(1, -1, -1),
                            autoDraw=True)
Probe = visual.GratingStim(win, tex="none", mask="circle", pos=(0, 0), size=(15, 15), color=(1, -1, -1), autoLog=True,
                           autoDraw=False)  # for center prob
MouseSpot = visual.GratingStim(win, tex="none", mask="circle", pos=(100, 100), size=(15, 15), color=(0, -1, 1),
                               autoLog=False)  # for mouse

# draw a ring with transparent center
Ring = visual.Circle(win, radius=EnvCfg.ctlsize, edges=64, fillColor=None, lineColor=(1, 1, 1), lineWidth=4,
                     autoLog=False, autoDraw=False)

# draw a arrow
GTArrow = visual.ShapeStim(
    win=win, name='RespArrow',
    vertices=[(-0.025, 0), (0.025, 0), (0.025, 0.45), (.1, 0.45), (0, 0.5), (-.1, 0.45), (-0.025, 0.45)],
    size=[1.0, 1.0],
    ori=1.0, pos=(0, 0),
    lineWidth=2.0, colorSpace='rgb', lineColor=(0.01, 0.9, 0.01), fillColor=(0.01, 0.9, 0.01),
    opacity=0.95, depth=-5.0, interpolate=True, autoDraw=False)

RespArrow = visual.ShapeStim(
    win=win, name='RespArrow',
    vertices=[(-0.025, 0), (0.025, 0), (0.025, 0.45), (.1, 0.45), (0, 0.5), (-.1, 0.45), (-0.025, 0.45)],
    size=[1.0, 1.0],
    ori=1.0, pos=(0, 0),
    lineWidth=2.0, colorSpace='rgb', lineColor=(1, 0.1, 0.1), fillColor=(1, 0.1, 0.1),
    opacity=0.95, depth=-6.0, interpolate=True, autoDraw=False)

message_arrow_res = visual.TextStim(win, pos=[0, 0], text='Your Response', color=(1, 0.1, 0.1), autoDraw=False)
message_arrow_gt = visual.TextStim(win, pos=[0, 0], text='Ground Truth', color=(0.01, 0.9, 0.01), autoDraw=False)

# this is for "credit card calibration"
stepsize = 5  # pixels
ang = 1  # here I want 50pixels =1VA
prepix = 50
physz = (8.56, 5.398)  # physcisal card size@cm
card = visual.Rect(win=win, name='polygon', width=428 * EnvCfg.ratM, height=270 * EnvCfg.ratM,
                   ori=0.0, pos=(0, 0), autoDraw='True',
                   lineWidth=1, colorSpace='rgb', lineColor='white', fillColor='white',
                   opacity=None, depth=0.0, interpolate=True, units='pix')

# generate image, 
# imagedir = os.path.join(pathPrefix, 'MOVZT', 'movie' + MOV)
imagedir = os.path.join(pathPrefix, 'MOVZT', 'test2')
# sin
oplist = list(np.sin(np.linspace(0 + .02, np.pi - .02, EnvCfg.number_of_frame)))

# create list empty list
imagelistO = [[] for _ in range(EnvCfg.modulation_num)]
imagelistH = [[] for _ in range(EnvCfg.modulation_num)]
imagelistV = [[] for _ in range(EnvCfg.modulation_num)]
imagelistHV = [[] for _ in range(EnvCfg.modulation_num)]

aperture = visual.Aperture(win, size=EnvCfg.aperture * 2 * EnvCfg.ratM)  # try shape='square'
aperture.enabled = False  # enabled by default when created

# create list empty list
imagelistO = [[] for _ in range(EnvCfg.modulation_num)]
imagelistH = [[] for _ in range(EnvCfg.modulation_num)]
imagelistV = [[] for _ in range(EnvCfg.modulation_num)]
imagelistHV = [[] for _ in range(EnvCfg.modulation_num)]

aperture = visual.Aperture(win, size=EnvCfg.aperture * 2 * EnvCfg.ratM)  # try shape='square'
aperture.enabled = False  # enabled by default when created
for types in range(1, EnvCfg.modulation_num + 1):  # 1,2,3,4...9
    print('loading image modulation type %d' % types)
    for i in range(EnvCfg.number_of_frame):
        imagepath = os.path.join(imagedir, 'type_%d' % types)
        imageInd = str("frame_%04d.png" % i)
        imagepath = os.path.join(imagepath, imageInd)
        imagelistO[types - 1].append(visual.ImageStim(win=win, image=imagepath,
                                                      size=(EnvCfg.image_size * EnvCfg.upampling_factor,
                                                            EnvCfg.image_size * EnvCfg.upampling_factor),
                                                      interpolate=True,
                                                      flipHoriz=False, flipVert=False))

        # horizontal flip the image sequence

        imagelistH[types - 1].append(visual.ImageStim(win=win, image=imagepath,
                                                      size=(EnvCfg.image_size * EnvCfg.upampling_factor,
                                                            EnvCfg.image_size * EnvCfg.upampling_factor),
                                                      interpolate=True, flipHoriz=True, flipVert=False))

        # vertical flip the image sequence
        imagelistV[types - 1].append(visual.ImageStim(win=win, image=imagepath,
                                                      size=(EnvCfg.image_size * EnvCfg.upampling_factor,
                                                            EnvCfg.image_size * EnvCfg.upampling_factor),
                                                      interpolate=True, flipHoriz=False, flipVert=True))
        # horizontal and vertical flip the image sequence
        imagelistHV[types - 1].append(visual.ImageStim(win=win, image=imagepath,
                                                       size=(EnvCfg.image_size * EnvCfg.upampling_factor,
                                                             EnvCfg.image_size * EnvCfg.upampling_factor),
                                                       interpolate=True, flipHoriz=True, flipVert=True))

print("All images have been loaded.")

# play credit card calibration
ok = 0
while not ok:
    # Listen for key presses until escape is pressed
    keys = kb.getKeys()
    if keys:
        if 'right' in keys:
            card.width += stepsize
        elif 'left' in keys:
            card.width -= stepsize
        elif 'up' in keys:
            card.height += stepsize
        elif 'down' in keys:
            card.height -= stepsize
        elif 'space' in keys:
            ok = 1
    adpix = [card.width, card.height]
    cmPpix = np.mean(np.divide(physz, adpix))
    resz = cmPpix * prepix
    dis = resz / (2 * np.tan(np.pi * ang / 360))
    message2.text = " Each pixel is %3.3f mm " % (cmPpix * 10)
    message3.text = " Please run formal experiment from distance: %3.0f cm" % (np.round(dis))
    win.flip()
card.autoDraw = False

# display instructions and wait
message1.text = 'Please press space key to start each trial.'
message2.text = 'You will see a short movie, please pay attention to center motion'
message3.text = 'please use mouse to refer direction and speed of the motion. Click mouse' \
                ' to confirm your response'
message4.text = 'if your mouse is not visible, please press right mouse button to make it visible'
message1.draw()
message2.draw()
message3.draw()
message4.draw()
win.flip()
# pause until there's a keypress
event.waitKeys()
win.mouseVisible = False
message1.autoDraw = False
message2.autoDraw = False
message3.autoDraw = False
message4.autoDraw = False

# Let's play formal Exp
# fliptimes = np.zeros(1500 + 1)
win.recordFrameIntervals = True
TotalframeN = 0
cc = 0
Trialcount = 1

for thisTrial in trials:  # handler can act like a for loop
    print('Trial %d / %d' % (Trialcount, EnvCfg.NumTrials))
    trialClock = core.Clock()
    # this is GT center
    problocx = thisTrial['GTX'] * EnvCfg.upampling_factor
    problocy = thisTrial['GTY'] * EnvCfg.upampling_factor
    mod_idx = int(thisTrial['Mod']) - 1
    # get RGB from image

    RGB = thisTrial['R'], thisTrial['G'], thisTrial['B']
    Probe1.color = (RGB[0], RGB[1], RGB[2])
    Probe2.color = (RGB[0], RGB[1], RGB[2])
    Probe3.color = (RGB[0], RGB[1], RGB[2])
    Probe4.color = (RGB[0], RGB[1], RGB[2])
    Probe.color = (RGB[0], RGB[1], RGB[2])

    picx = (imagelistO[mod_idx][0].size[0] / 2) - problocx
    picy = problocy - (imagelistO[mod_idx][0].size[1] / 2)

    Probe.draw()
    Ring.opacity = .10
    Ring.autoDraw = True
    win.flip()
    GTA = thisTrial['Angle']
    GTR = thisTrial['Radius'] * EnvCfg.upampling_factor  # pixel/frame

    # Note: you need calculate the true grond truth in advance
    if thisTrial['FT'] == 1:
        tmpimagelist = imagelistO[mod_idx]
    elif thisTrial['FT'] == 2:  # horizontal flip
        tmpimagelist = imagelistH[mod_idx]
        picx = -picx
        GTA = np.arctan2(np.sin(GTA), -np.cos(GTA))
    elif thisTrial['FT'] == 3:  # vertical flip
        tmpimagelist = imagelistV[mod_idx]
        picy = -picy
        GTA = np.arctan2(-np.sin(GTA), np.cos(GTA))
    elif thisTrial['FT'] == 4:
        tmpimagelist = imagelistHV[mod_idx]
        GTA = np.arctan2(-np.sin(GTA), -np.cos(GTA))
        picx = -picx
        picy = -picy
    else:
        print('Error: wrong flip type')
    if GTA < 0:
        GTA = GTA + 2 * np.pi
    GTA = GTA * 180 / np.pi

    # import matplotlib.pyplot as plt
    # plt.imshow(mask)
    # plt.show()
    picloc = [picx, picy]

    for ii in range(EnvCfg.number_of_frame):
        tmpimagelist[ii].setPos([picloc[0], picloc[1]])
        tmpimagelist[ii].opacity = oplist[ii]

    # Let's start movie (& response)!
    aperture.setPos((0, 0))
    aperture.enabled = True
    myMouse.setPos(newPos=(50, 50))
    mouseclick = 0
    framecount = 0
    repeatcount = 0
    # get current refresh rate
    refresh_rate = win.getActualFrameRate()
    while mouseclick == 0 or repeatcount < 20:
        if refresh_rate > 1.2 * EnvCfg.frame_rate:
            duri = 1.0 / EnvCfg.frame_rate
            diff = duri - frame_dur
            if diff > 0:
                core.wait(diff)  # synchronize to 30 frame per second

        ctrI = (framecount % 90)
        # MPI(0-30), ISI(31-60), pink(60-80),ISI(80-100). using 90 frames as a cycle to control procedure
        # mouse stuffs
        mouse_dX, mouse_dY = myMouse.getPos()
        MouseSpot.setPos([mouse_dX, mouse_dY])
        delta = (mouse_dX - ConC[0]), (mouse_dY - ConC[1])
        ang = np.arctan2(delta[1], delta[0])
        if np.sqrt(delta[0] ** 2 + delta[1] ** 2) <= Ctlsize:
            # print(np.sqrt(delta[0] ** 2 + delta[1] ** 2))
            vel = (2 ** ((np.sqrt(delta[0] ** 2 + delta[
                1] ** 2) / Ctlsize) * CtlMaxSp)) - 1  # CtlMaxSp control panel for is  25-1 pixels/frame

            response_dX = vel * np.cos(ang)
            response_dY = vel * np.sin(ang)
        else:  # if faster then maxium speed, then maxium speed, i.e. 24 pixels/frame
            vel = ((2 ** CtlMaxSp) - 1)
            response_dX = vel * np.cos(ang)
            response_dY = vel * np.sin(ang)
        ang = ang * 180 / np.pi
        if delta[1] < 0:  # if y is negative, then polar will become negative so +360
            ang = ang + 360
        noise.phase += (response_dX / noisesize[0], response_dY / noisesize[1])
        mouse1, mouse2, mouse3 = myMouse.getPressed()
        # print(vel)
        # print(ang)

        # # for debug error
        # RespArrow.setOri(-(ang-90))
        # RespArrow.setSize((100, vel/20*600))
        # # # draw arrow
        # RespArrow.draw()
        if ctrI == 0:  # onset of movie
            win.recordFrameIntervals = True
        elif ctrI == 50:  # onset of response
            noise.opacity = .02
            #
        if ctrI <= EnvCfg.number_of_frame - 1:  # play movie
            tmpimagelist[ctrI].draw()
        elif 50 <= ctrI <= 80:  # play response
            noise.draw()
        if (EnvCfg.probe_onset_frame - 2) <= ctrI <= (EnvCfg.probe_onset_frame + 3):
            # show prob  to refer timing, currently 5 frames
            Probe.draw()
        MouseSpot.draw()
        win.logOnFlip(msg='frame=%i' % TotalframeN, level=logging.EXP)
        win.flip()  # refresh screen
        framecount += 1

        if mouse1:  # this for response stop
            if repeatcount >= 2:  # wait for 2 cycles to make sure the response is stable
                mouseclick = 1
                # set arrow vector to [mouse_dX, mouse_dY]
                mouse_dX, mouse_dY = 0, Ctlsize + 100
                break
                # detect whether press the right mouse button
        if mouse3:  # set mouse position to center
            myMouse.setPos(newPos=(20, 20))
        # this control the opacity of movie and response
        if 50 <= ctrI <= 60:
            noise.opacity += .1
            # Ring.opacity -= .05

        elif 60 <= ctrI <= 70:
            noise.opacity -= .1
            # Ring.opacity += .05
        if ctrI == 90 - 1:
            repeatcount += 1
            win.recordFrameIntervals = False

        if repeatcount == 20:  # if no response, no ang/vel, get out!
            if mouseclick == 0:
                ang = np.NAN
                vel = np.NAN
                break

    # response finish, put some feedback
    aperture.enabled = False
    noise.opacity = 0.0
    Ring.opacity = 1.0
    Ring.draw()
    # GT

    MouseSpot.setPos([mouse_dX, mouse_dY])
    if EnvCfg.feedback:
        message1.text = "Trial: %03.f / %03.d" % (Trialcount, EnvCfg.NumTrials)
        message2.text = f"GT:  %.3f° with %.3f pix/frame" % (GTA, GTR)
        message3.text = "Response:   %.3f° with %.3f pix/frame" % (ang, vel)

        # adjust the position of the text
        message1.setPos([0, EnvCfg.aperture + 200])
        message2.setPos([0, EnvCfg.aperture + 150])
        message3.setPos([0, EnvCfg.aperture + 100])

        # transfer angle/speed to spatial location

        GTArrow.setSize((100, GTR * EnvCfg.ctlsize * 2 / 20))
        GTArrow.setOri(-(GTA - 90))
        RespArrow.setSize((100, vel * EnvCfg.ctlsize * 2 / 20))
        RespArrow.setOri(-(ang - 90))

        # trasnfer angle/speed to spatial location
        RespX = np.cos(ang * np.pi / 180) * (vel + 2) * EnvCfg.ctlsize / 20
        RespY = np.sin(ang * np.pi / 180) * (vel + 2) * EnvCfg.ctlsize / 20
        GTX = np.cos(GTA * np.pi / 180) * (GTR + 2) * EnvCfg.ctlsize / 20
        GTY = np.sin(GTA * np.pi / 180) * (GTR + 2) * EnvCfg.ctlsize / 20

        message_arrow_res.setPos([RespX, RespY])
        message_arrow_gt.setPos([GTX, GTY])
        message_arrow_gt.setOri(-(GTA - 90))
        message_arrow_res.setOri(-(ang - 90))
        # draw arrow
        message_arrow_res.draw()
        message_arrow_gt.draw()
        GTArrow.draw()
        RespArrow.draw()
        message1.draw()
        message2.draw()
        message3.draw()
        noise.draw()
    else:
        message1.text = "Trial: %03.f" % Trialcount
        message3.text = "Response:   %.3f° with %.3f pix/frame" % (ang, vel)

        # adjust the position of the text
        message1.setPos([0, EnvCfg.aperture + 150])
        message3.setPos([0, EnvCfg.aperture + 100])

        # transfer angle/speed to spatial location

        GTArrow.setSize((100, GTR * EnvCfg.ctlsize * 2 / 20))
        GTArrow.setOri(-(GTA - 90))
        RespArrow.setSize((100, vel * EnvCfg.ctlsize * 2 / 20))
        RespArrow.setOri(-(ang - 90))

        # trasnfer angle/speed to spatial location
        RespX = np.cos(ang * np.pi / 180) * (vel + 2) * EnvCfg.ctlsize / 20
        RespY = np.sin(ang * np.pi / 180) * (vel + 2) * EnvCfg.ctlsize / 20
        GTX = np.cos(GTA * np.pi / 180) * (GTR + 2) * EnvCfg.ctlsize / 20
        GTY = np.sin(GTA * np.pi / 180) * (GTR + 2) * EnvCfg.ctlsize / 20
        message_arrow_res.setPos([RespX, RespY])
        message_arrow_gt.setPos([GTX, GTY])
        message_arrow_gt.setOri(-(GTA - 90))
        message_arrow_res.setOri(-(ang - 90))
        # draw arrow
        message_arrow_res.draw()
        RespArrow.draw()
        message1.draw()
        message3.draw()
        noise.draw()

    win.flip()

    trials.data.add('GTvelocity', GTR)
    trials.data.add('GTAngle', GTA)
    trials.data.add('RAngle', ang)
    trials.data.add('Rvelocity', vel)  # note here is pixels/frame,
    trials.data.add('picX', picloc[0])  # this is the picture center
    trials.data.add('picY', picloc[1])
    trials.data.add('Reptime', repeatcount)
    Trialcount += 1

    # wait a keypress for next trial or end
    feedbackend = None
    while feedbackend is None:
        allKeys = event.waitKeys()
        for thisKey in allKeys:
            if (thisKey == 'space'):
                feedbackend = 1
            elif thisKey in ['q', 'escape']:
                core.quit()

            # give some on-screen feedback
Endthank = visual.TextStim(
    win, pos=[0, +3], color=(1, 1, 1),
    text='Thank you! You have finished the experiment!')
Endthank.draw()
win.flip()
event.waitKeys()  # wait for participant to respond

# Write trials data to a csv ...
fileName = 'MPIData_' + fileName + '.csv'
trials.saveAsExcel(fileName, sheetName='rawData')

# Wide format is useful for analysis with R or SPSS.
df = trials.saveAsWideText('MPIData_' + fileName + '.txt')
# wirte the data to a csv file
df.to_csv('MPIData_' + fileName + '.csv')

gc.enable()
core.rush(False)
win.close()

"""do something to check timing"""
"""import matplotlib
matplotlib.use('Qt4Agg')  # change this to control the plotting 'back end'
import pylab


# calculate some values
intervalsMS = pylab.array(win.frameIntervals) * 1000
m = pylab.mean(intervalsMS)
sd = pylab.std(intervalsMS)
# se=sd/pylab.sqrt(len(intervalsMS)) # for CI of the mean

msg = "Mean=%.2fms, s.d.=%.2f, 99%%CI(frame)=%.2f-%.2f"
distString = msg % (m, sd, m - 2.58 * sd, m + 2.58 * sd)
nTotal = len(intervalsMS)
nDropped = sum(intervalsMS > (1.5 * m))
msg = "Dropped/Frames = %i/%i = %.3f%%"
droppedString = msg % (nDropped, nTotal, 100 * nDropped / float(nTotal))

# plot the frameintervals
pylab.figure(figsize=[12, 8])
pylab.subplot(1, 2, 1)
pylab.plot(intervalsMS, '-')
pylab.ylabel('t (ms)')
pylab.xlabel('frame N')
pylab.title(droppedString)

pylab.subplot(1, 2, 2)
pylab.hist(intervalsMS, 50, histtype='stepfilled')
pylab.xlabel('t (ms)')
pylab.ylabel('n frames')
#pylab.﻿ylim(0, 360)

pylab.title(distString)
pylab.show()
"""

core.quit()
