Directory:

Root
|-final_project.py
|-stabilization.py
|-/videos
     |-/output
     |-/source
          |-*.mov
          |-*.mp4
          |-*.avi

Required libraries:
Use the CS6475 conda environment:
$conda activate CS6475
Update numpy:
$pip install numpy --upgrade
Install cvxpy:
$conda install cvxpy

Instructions:
Setup root directory as shown.
Input videos must be located in relative path ./videos/source
Valid extensions are .mov, .mp4, .avi (more can be added to the EXT variable in final_project.py header)
Source and output folders are stored in SOURCE and OUTPUT in final_project.py header
run $python final_project.py
It will automatically stabilize each video in the source folder.

Video Link: https://gatech.box.com/s/6mll6dvbyaz783z81xui83tmyvjot69b