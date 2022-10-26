# LRS2Multi

## Setup for TACC users
To set up LRS2Multi on TACC you need the following four commands to run "successfully."
```
ssh username@stampede2.tacc.utexas.edu
~ecooper/bin/tacc_setup
cd /work/NUMBER/NAME/stampede2
git clone https://github.com/grzeimann/LRS2Multi.git
```
Then, go to the visualization portal in a browser: https://vis.tacc.utexas.edu/jobs/

<p align="center">
  <img src="TACC_VIZ_portal.png" width="650"/>
</p>

Request a job as shown in the attached image (just click submit when you pull up the same left hand settings). After a small wait time, a new screen will show up and you will click connect.  Sometimes there are not enough nodes initially and you have to wait a bit longer. After you connect, you should be in your work directory, which will allow you to navigate to LRS2Multi/notebooks.  There will be a file called example_reduction.ipynb.  Open that notebook and follow the instructions to get started.
