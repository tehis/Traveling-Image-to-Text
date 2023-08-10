#!/bin/bash
mlflow server --host 0.0.0.0 &


python app/app.py


fg %1
