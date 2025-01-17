from psychopy import visual, core, event

win = visual.Window(size=(800, 600), fullscr=False, allowGUI=True)

RespArrow = visual.ShapeStim(
    win=win, vertices=[(-0.025, 0), (0.025, 0), (0.025, 0.45), (.1, 0.45), (0, 0.5), (-.1, 0.45), (-0.025, 0.45)],
    lineWidth=2.0, interpolate=True, closeShape=True,fillColor=(1,1,1), lineColor=(1,0,0), opacity=1.0, depth=0.0,
    pos=(0, 0), size=(1, 3), ori=60.0, contrast=1.0,  autoDraw=False)

RespArrow.setOri(60)
RespArrow.setVertices([(-0.025, 0), (0.025, 0), (0.025, 0.45), (.1, 0.45), (0, 0.5), (-.1, 0.45), (-0.025, 0.45)])

RespArrow.draw()
win.flip()

event.waitKeys()

win.close()
