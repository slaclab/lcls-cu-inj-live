# lcls-cu-inj-live

This project contains the tools for running the live online model for the lcls-cu injector. The model is served over EPICS and the client uses bokeh for web-based rendering.

## Environment

The environment for this project may be set up using conda using the included `environment.yml`:

```
$ conda env create -f environment.yml
```

Once complete, activate the environment:

```
$ conda activate lcls-cu-inj-live
```

If using custom classes or models defined in the repository, the repository root must be included in the PYTHONPATH.

From repostitory root:

```
$ export PYTHONPATH=$(pwd)
```

## Server

Serve the client using the test prefix:
```
$ serve-from-template files/model_config.yml test
```

## Client

The client for this demo may be launched either using the tools packaged with `lume-epics` for auto client generation or with the custom built client. Using the `lume-epics` tools, the client may be launched using the command:

```
$ render-from-template {FILENAME} {PROTOCOL} {PREFIX} --read-only
```
The optional read-only flag determines whether the client will be launched with controls. Clients rendered in read-only mode will also produce a striptool per scalar variable as opposed to a striptool with the selection option.

To launch the display using a Channel Access client:

```
$ render-from-template files/model_config.yml ca test
```

The pre-built read only client may be launched using:

```
$ bokeh serve app --args {PREFIX} {PROTOCOL}
```

## Instructions to run at LCLS

As of now, it is recommended to run everything from mcc-simul.

```shell script
ssh mcc-simul
# If not yet checked out:
git clone https://github.com/slaclab/lcls-cu-inj-live.git
cd lcls-cu-inj-live
# Start the model server and PV bridge
./start_all &
```

### If running the model server outside of mcc-simul

- Edit start_webserver, change line 6 and replace `mcc-simul` with the hostname
with the proper one for the model server so the web server can reach the PVs.
