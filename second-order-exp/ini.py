from psychopy.tools.filetools import fromFile, toFile
from psychopy.hardware import keyboard
from configure import get_config
from builtins import range
import pandas as pd
import gc
import matplotlib
from psychopy import core, visual, gui, data, event, monitors
import os


def initialize():
    EnvCfg = get_config()
    matplotlib.use('Qt5Agg')  # change this to control the plotting 'back end'
    expInfo = {'Observer': '01', 'Block': '01'}

    expInfo['dateStr'] = data.getDateStr()  # add the current time

    # present a dialogue to change params

    dlg = gui.DlgFromDict(expInfo, title='MPI HS FlipGrid', fixed=['dateStr'])
    if not dlg.OK:
        core.quit()  # the user hit cancel so exit

    Block = int(expInfo['Block'])
    if not Block == EnvCfg.TrailBlock:
        print('Warning! Block number is not correct! ')
        EnvCfg.TrailBlock = Block
        print('Block number is automatically set to %s' % Block)
    # import condition list from excel
    file = EnvCfg.TrailXlsxFile
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
    Trail_Block = pdSetupFile['Trail_Block'].tolist()

    # extract the trails index with Trail_Block = EnvCfg.TrailBlock
    print("Conducting Block %d" % EnvCfg.TrailBlock)
    MovI = [MovI[i] for i in range(len(Trail_Block)) if Trail_Block[i] == EnvCfg.TrailBlock]
    GTX = [GTX[i] for i in range(len(Trail_Block)) if Trail_Block[i] == EnvCfg.TrailBlock]
    GTY = [GTY[i] for i in range(len(Trail_Block)) if Trail_Block[i] == EnvCfg.TrailBlock]
    Angle = [Angle[i] for i in range(len(Trail_Block)) if Trail_Block[i] == EnvCfg.TrailBlock]
    Radius = [Radius[i] for i in range(len(Trail_Block)) if Trail_Block[i] == EnvCfg.TrailBlock]
    flowU = [flowU[i] for i in range(len(Trail_Block)) if Trail_Block[i] == EnvCfg.TrailBlock]
    flowV = [flowV[i] for i in range(len(Trail_Block)) if Trail_Block[i] == EnvCfg.TrailBlock]
    flipType = [flipType[i] for i in range(len(Trail_Block)) if Trail_Block[i] == EnvCfg.TrailBlock]
    R = [R[i] for i in range(len(Trail_Block)) if Trail_Block[i] == EnvCfg.TrailBlock]
    G = [G[i] for i in range(len(Trail_Block)) if Trail_Block[i] == EnvCfg.TrailBlock]
    B = [B[i] for i in range(len(Trail_Block)) if Trail_Block[i] == EnvCfg.TrailBlock]
    Modulation = [Modulation[i] for i in range(len(Trail_Block)) if Trail_Block[i] == EnvCfg.TrailBlock]
    print("Total %d trials" % len(MovI))

    assert len(MovI) == EnvCfg.NumTrials, "The number of trials is not correct!"

    max_speed = max(Radius) * EnvCfg.upampling_factor
    # making stimuli
    stimList = []
    for ctr in range(0, EnvCfg.NumTrials):  # TODO 25space*3time*4flip
        stimList.append({'mov': MovI[ctr], 'Mod': Modulation[ctr], 'GTX': GTX[ctr], 'GTY': GTY[ctr],
                         'Angle': Angle[ctr], 'Radius': Radius[ctr], 'flowU': flowU[ctr], 'flowV': flowV[ctr],
                         'FT': flipType[ctr],
                         'R': R[ctr],
                         'G': G[ctr], 'B': B[ctr], 'Trail_Block': EnvCfg.TrailBlock})

    # organize them with the trial handler
    trials = data.TrialHandler(stimList, 1, method='sequential',  # random or sequential
                               extraInfo={'participant': expInfo['Observer'], 'Block': expInfo['Block']})

    win = visual.Window(allowGUI=True, color=(0, 0, 0), allowStencil=True, monitor='testMonitor', winType='pyglet',
                        units='pix', screen=1, fullscr=True)
    # screen=1 means the second monitor

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

    # load image:
    # create list empty list
    image_list = [[] for _ in range(EnvCfg.NumTrials)]

    aperture = visual.Aperture(win, size=EnvCfg.aperture * 2 * EnvCfg.ratM)  # try shape='square'
    aperture.enabled = False  # enabled by default when created
    idx = 0
    for scene_id, mod_type, flipType in zip(MovI, Modulation, flipType):
        print("loading scene %d, modulation %d, flip type %d. (%d,%d)" % (scene_id, mod_type, flipType, idx, len(MovI)))
        for i in range(EnvCfg.number_of_frame):
            imagepath = os.path.join(EnvCfg.img_root, 'Scene_%d\\' % scene_id + 'type_%d' % mod_type)
            imageInd = str("frame_%04d.png" % i)
            imagepath = os.path.join(imagepath, imageInd)
            if flipType == 1:
                image_list[idx].append(visual.ImageStim(win=win, image=imagepath,
                                                        size=(EnvCfg.image_size * EnvCfg.upampling_factor,
                                                              EnvCfg.image_size * EnvCfg.upampling_factor),
                                                        interpolate=True,
                                                        flipHoriz=False, flipVert=False))
            elif flipType == 2:
                image_list[idx].append(visual.ImageStim(win=win, image=imagepath,
                                                        size=(EnvCfg.image_size * EnvCfg.upampling_factor,
                                                              EnvCfg.image_size * EnvCfg.upampling_factor),
                                                        interpolate=True,
                                                        flipHoriz=True, flipVert=False))
            elif flipType == 3:
                image_list[idx].append(visual.ImageStim(win=win, image=imagepath,
                                                        size=(EnvCfg.image_size * EnvCfg.upampling_factor,
                                                              EnvCfg.image_size * EnvCfg.upampling_factor),
                                                        interpolate=True,
                                                        flipHoriz=False, flipVert=True))
            elif flipType == 4:
                image_list[idx].append(visual.ImageStim(win=win, image=imagepath,
                                                        size=(EnvCfg.image_size * EnvCfg.upampling_factor,
                                                              EnvCfg.image_size * EnvCfg.upampling_factor),
                                                        interpolate=True,
                                                        flipHoriz=True, flipVert=True))
            else:
                raise ValueError("flip type is not correct!")

        idx += 1

    return win, EnvCfg, stimList, myMouse, globalClock, kb, frame_dur, refresh_rate, Block, trials, expInfo, aperture, \
        image_list, max_speed


if __name__ == '__main__':
    A = \
        initialize()
