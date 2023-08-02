#!/bin/bash

conda deactivate
tmux new-session -s "dod" -n "root" "tmux source-file tmux/session"
