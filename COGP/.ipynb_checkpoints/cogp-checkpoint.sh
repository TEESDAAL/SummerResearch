#!/bin/sh
#
# Force Bourne Shell if not Sun Grid Engine default shell (you never know!)
#
#$ -S /bin/sh
#
# I know I have a directory here so I'll use it as my initial working directory
#
#$ -wd /vol/grid-solar/sgeusers/teesdaalan 
#
#
# End of the setup directives
#
# Now let's do something useful, but first change into the job-specific
# directory that should have been created for us
#
# Check we have somewhere to work now and if we don't, exit nicely.
#
if [ -d /local/tmp/teesdaalan/$JOB_ID.$SGE_TASK_ID ]; then
        cd /local/tmp/teesdaalan/$JOB_ID.$SGE_TASK_ID
else
        echo "Uh oh ! There's no job directory to change into "
        echo "Something is broken. I should inform the programmers"
        echo "Save some information that may be of use to them"
        echo "Here's LOCAL TMP "
        ls -la /local/tmp
        echo "AND LOCAL TMP TEESDAALAN "
        ls -la /local/tmp/teesdaalan
        echo "Exiting"
        exit 1
fi

model=$1
homeDir='/vol/grid-solar/sgeusers/teesdaalan/SummerResearch/COGP'

cd $homeDir

python main.py -s $SGE_TASK_ID -p 50 -g 50
