# TasNet: Time-domain Audio Separation Network
A PyTorch implementation of "TasNet: Time-domain Audio Separation Network for Real-time, single-channel speech separation", published in ICASSP2018, by Yi Luo and Nima Mesgarani.

## Results
| Method | Causal | SDRi | SI-SNRi | Config |
| :----: | :----: | :-----: | :--: | :----: |
| TasNet-BLSTM (Paper) | No | 11.1 | 10.8 | |
| TasNet-BLSTM (Here) | No | 11.84 | 11.54 | L40 N500 hidden500 layer4 lr1e-3 epoch100 batch size10 |
| TasNet-BLSTM (Here) | No | 11.77 | 11.46 | + L2 1e-4|
| TasNet-BLSTM (Here) | No | 13.07 | 12.78 | + L2 1e-5|

## Install
- PyTorch 0.4.1+
- Python3 (Recommend Anaconda)
- `pip install -r requirements.txt`
- If you need to convert wjs0 to wav format and generate mixture files, `cd tools; make`

## Usage
If you already have mixture wsj0 data:
1. `$ cd egs/wsj0`, modify wsj0 data path `data` to your path in the beginning of `run.sh`.
2. `$ bash run.sh`, that's all!

If you just have origin wsj0 data (sphere format):
1. `$ cd egs/wsj0`, modify three wsj0 data path to your path in the beginning of `run.sh`.
2. Convert sphere format wsj0 to wav format and generate mixture. `Stage 0` part provides an example.
3. `$ bash run.sh`, that's all!

You can change hyper-parameter by `$ bash run.sh --parameter_name parameter_value`, egs, `$ bash run.sh --stage 3`. See parameter name in `egs/aishell/run.sh` before `. utils/parse_options.sh`.
### Workflow
Workflow of `egs/wsj0/run.sh`:
- Stage 0: Convert sphere format to wav format and generate mixture (optional)
- Stage 1: Generating json files including wav path and duration
- Stage 2: Training
- Stage 3: Evaluate separation performance
- Stage 4: Separate speech using TasNet
### More detail
```bash
# Set PATH and PYTHONPATH
$ cd egs/wsj0/; . ./path.sh
# Train:
$ train.py -h
# Evaluate performance:
$ evaluate.py -h
# Separate mixture audio:
$ separate.py -h
```
#### How to visualize loss?
If you want to visualize your loss, you can use [visdom](https://github.com/facebookresearch/visdom) to do that:
1. Open a new terminal in your remote server (recommend tmux) and run `$ visdom`
2. Open a new terminal and run `$ bash run.sh --visdom 1 --visdom_id "<any-string>"` or `$ train.py ... --visdom 1 --vidsdom_id "<any-string>"`
3. Open your browser and type `<your-remote-server-ip>:8097`, egs, `127.0.0.1:8097`
4. In visdom website, chose `<any-string>` in `Environment` to see your loss
#### How to resume training?
```bash
$ bash run.sh --continue_from <model-path>
```
## TODO
- [ ] Layer normlization described in paper
- [ ] LSTM skip connection
- [ ] Curriculum learning
