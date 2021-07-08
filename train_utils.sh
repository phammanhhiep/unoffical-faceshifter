#!/bin/bash

_logger() {
    echo "$(date '+%d%m%Y_%H%M%S'): $1"
}

is_training() {
    ps aux | egrep "aei_trainer.py" | grep -q -v "grep"
    return $?
}

restart_train_process() {
    if ! is_training; then
        _logger "No train process is found"
        cd $HOME/projects/ufaceshifter
        export PYTHONPATH=$HOME/projects/pytorch-lightning
        $HOME/projects/ufaceshifter/train.sh artifacts/checkpoints/other_research_code/last.ckpt
    fi
}

restart_train_process