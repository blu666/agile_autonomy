# Setting Up the Agile Autonomy Environment

## Prerequisites
    Ubuntu 20.04
    ROS-noetic
    Anaconda
    gcc/g++ 7.5.0


## Procedures

    sudo apt-get update
    sudo apt-get install git cmake g++

### Install catkin
    sudo apt-get install ros-noetic-catkin python3-catkin-tools

### If ROS-noetic wasn't already installed, 
    # Install ROS Noetic
    # Don't forget to configure your Ubuntu repositories to allow "restricted," "universe," and "multiverse."
    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
    sudo apt install curl # if you haven't already installed curl
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
    sudo apt update
    sudo apt install ros-noetic-desktop-full
    source /opt/ros/noetic/setup.bash
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
    source ~/.bashrc
    sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
    sudo apt install python3-rosdep
    sudo rosdep init
    rosdep update


### Make Directory
    export ROS_VERSION=noetic
    mkdir aa_ws
    cd aa_ws

### Install Open3D
    git clone --recursive https://github.com/intel-isl/Open3D
    cd Open3D
    git checkout v0.9.0
    git submodule update --init --recursive
    ./util/scripts/install-deps-ubuntu.sh
    mkdir build
    cd build

    cmake ..
    make -j$(nproc)
    sudo make install
Note that running "cmake .." instead of "cmake -DCMAKE_INSTALL_PREFIX=/usr/local/bin/cmake .." will avoid issues with open3d

### Now we can install agile autonomy
#### First need to get old compilers
    sudo apt-get install g++-7 gcc-7 # agile autonomy requires old compilers for some reason
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 100
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 100

#### Setup catkin workspace and clone git repo
    export CATKIN_WS=./catkin_aa
    mkdir -p $CATKIN_WS/src
    cd $CATKIN_WS
    catkin init
    catkin config --extend /opt/ros/$ROS_VERSION
    catkin config --merge-devel
    catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-fdiagnostics-color -DPYTHON_EXECUTABLE:FILEPATH=/usr/bin/python3
    cd src
    git clone git@github.com:uzh-rpg/agile_autonomy.git [or git@github.com:blu666/agile_autonomy.git]

Install any necessary packages

    sudo catkin build
### Important Issues

1. Need to comment out line 39 in catkin_aa/src/rpg_mpl_ros/planning_ros_utils/src/planning_rviz_plugins/map_display.cpp

    Or alterantively, you can change line 39 to 

        update_nh_.setCallbackQueue(context_->getThreadedQueue());

2. If have cv_bridge error with "importError: dynamic module does not define module export function (PyInit_cv_bridge_boost)"

    - If only occurs in Anaconda environment, run:
        
        conda install -c conda-forge ros-cv-bridge
    - Else, may need to build cv_bridge from source
        - https://shliang.blog.csdn.net/article/details/117230220
        - https://blog.csdn.net/wxm__/article/details/121220816


### Potential Problems I ran into
    - Problems with "roslaunch * *.launch"
      - If running a conda environment, even if in base environment, you need to exit the conda environment. 
        - run "conda deactivate"
      - Problems with "roslaunch flightros rotors_gazebo.launch"
        - This may be caused by some interaction with Anaconda.
        - Fix: run "conda install -c conda forge rospkg

    - Problems with running with warning message "sh 1: rosservice not found"
        - Script isn't able to call rosservice gazebo/pause_physics and gazebo/unpause_physics correctly. And simulation will get stuck after a few runs.
        - Fixed by uninstalling and reinstalling ros and rosservice pacakges (only reinstall rosservice might work)
            "sudo apt-get install python3-rosservice"
            "sudo apt-get install ros-noetic-rosservice"



## References
https://github.com/uzh-rpg/agile_autonomy/issues/10