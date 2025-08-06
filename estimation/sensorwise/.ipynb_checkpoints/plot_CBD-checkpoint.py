# 필요한 함수 정의
import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt

def meanStdPlot(position,data,name):
    color_X = "#008080"##015734
    color_Y = "#4d019a"#58008f
    linewidth = 2
    
    subAJ_right = 0.5
    subAJ_top = 1.0
    subAJ_wspace = 0.5
    sub_offset= 40

    data = np.array(data).astype(float)
    print('data : ',data.shape)
    
    host = host_subplot(position, axes_class=AA.Axes)
    
    plt.subplots_adjust(right=subAJ_right,top=subAJ_top, wspace=subAJ_wspace)

    par1 = host.twinx()

    par1.axis["right"].toggle(all=True) 


    # host.set_xlim(0, 2)
    # host.set_ylim(0, 2)

    host.set_xlabel(f"Normalized Time")
    host.set_title(
    f"{name}"
    )
    # host.set_ylabel("Angle")
    # par1.set_ylabel("Angle")
    # par2.set_ylabel("Z_axis")

    p1, = host.plot(range(0,len(data[0])), data[0], linewidth=linewidth,label="True",color=color_X)
    host.fill_between(range(0,len(data[0])), data[0]+data[1],data[0]-data[1],alpha=0.2,facecolor=p1.get_color(), edgecolor=(0,0,0,.8))
    p2, = par1.plot(range(0,len(data[0])),  data[2],linewidth=linewidth, label="Pred",color=color_Y)
    par1.fill_between(range(0,len(data[0])), data[2]+data[3],data[2]-data[3],alpha=0.2,facecolor=p2.get_color(),edgecolor=(0,0,0,.8))
    
    if 'angle' in name:
        if 'X' in name:
            host.set_ylim(-55, 0)
            par1.set_ylim(-55, 0)
        elif 'Y' in name:
            host.set_ylim(-10, 10)
            par1.set_ylim(-10, 10)
        elif 'Z' in name:   
            host.set_ylim(-25, 10)
            par1.set_ylim(-25, 10)

    # elif 'moBWHT' in name:
    #     host.set_ylim(0, 4)
    #     par1.set_ylim(0, 4)
     

    host.legend()

    host.axis["left"].label.set_color(color_X)
    par1.axis["right"].label.set_color(color_Y)


    host.axis["left"].major_ticks.set_color(color_X)
    par1.axis["right"].major_ticks.set_color(color_Y)


    host.axis["left"].major_ticklabels.set_color(color_X)
    par1.axis["right"].major_ticklabels.set_color(color_Y)

def diffPlot(position,data,name):
    if "X" in name:
        color_X = "#e05858"
    elif "Y" in name:
        color_X = "#32a852"
    elif "Z" in name:
        color_X = "#4159cc"
    linewidth = 2
    
    subAJ_right = 0.5
    subAJ_top = 1.0
    subAJ_wspace = 0.5
    sub_offset= 40

    data = np.array(data).astype(float)
    print('data : ',data.shape)
    
    host = host_subplot(position, axes_class=AA.Axes)
    
    plt.subplots_adjust(right=subAJ_right,top=subAJ_top, wspace=subAJ_wspace)

    # host.set_xlim(0, 2)
    # host.set_ylim(0, 2)

    host.set_xlabel(f"Normalized Time")
    host.set_title(
    f"{name}"
    )
    # host.set_ylabel("Angle")
    # par1.set_ylabel("Angle")
    # par2.set_ylabel("Z_axis")

    p1, = host.plot(range(0,len(data[0])), data[0], linewidth=linewidth,label="Diff",color=color_X)
    host.fill_between(range(0,len(data[0])), data[0]+data[1],data[0]-data[1],alpha=0.2,facecolor=p1.get_color(), edgecolor=(0,0,0,.8))

    if 'angle' in name:
        host.set_ylim(0, 15)
    elif 'moBWHT' in name:
        host.set_ylim(0, 5)

     

    host.legend()

    host.axis["left"].label.set_color(color_X)

    host.axis["left"].major_ticks.set_color(color_X)

    host.axis["left"].major_ticklabels.set_color(color_X)