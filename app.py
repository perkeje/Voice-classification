import streamlit as st
import torch
import torchaudio
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torch import nn

import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetVoiceClassifier(nn.Module):
    def __init__(self, block, layers, num_classes=251):
        super(ResNetVoiceClassifier, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x


def resnet18(num_classes):
    return ResNetVoiceClassifier(BasicBlock, [2, 2, 2, 2], num_classes)


# Function to make a prediction
def predict(model, audio, device):
    model.eval()
    with torch.no_grad():
        audio = audio.unsqueeze(0)
        output = model(audio)
        prediction = torch.argmax(output[0])
    return prediction


def create_speaker_map():
    speaker_map = {}
    new_id = 0
    num_lines_to_skip = 12

    with open("./SPEAKERS.TXT", "r") as f:
        # Skip the header line
        for _ in range(num_lines_to_skip):
            next(f)
        for line in f:
            reader_id, gender, subset, minutes, name, *_ = line.strip().split("|")
            if subset.strip() == "train-clean-100":
                speaker_map[new_id] = name
                new_id += 1
            if new_id >= 10:  # Stop after mapping 100 speakers
                break
    return speaker_map


def load_and_preprocess(signal, sr, transformations, num_samples, target_sr, device):
    signal = _resample_if_necessary(signal, sr, target_sr)
    signal = _mix_down_if_necessary(signal)
    signal = _cut_if_necessary(signal, num_samples)
    signal = _right_pad_if_necessary(signal, num_samples)
    signal = signal.to(device)
    signal = transformations(signal)
    signal = _normalize_global(signal)
    return signal


def _cut_if_necessary(signal, num_samples):
    if signal.shape[1] > num_samples:
        midpoint = signal.shape[1] // 2
        start = midpoint - (num_samples // 2)
        end = start + num_samples
        if start < 0:
            start = 0
            end = num_samples
        elif end > signal.shape[1]:
            end = signal.shape[1]
            start = end - num_samples
        signal = signal[:, start:end]
    return signal


def _right_pad_if_necessary(signal, num_samples):
    length_signal = signal.shape[1]
    if length_signal < num_samples:
        num_missing_samples = num_samples - length_signal
        last_dim_padding = (0, num_missing_samples)
        signal = torch.nn.functional.pad(signal, last_dim_padding)
    return signal


def _resample_if_necessary(signal, sr, target_sr):
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        signal = resampler(signal)
    return signal


def _mix_down_if_necessary(signal):
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal


def _normalize_global(signal):
    mean = signal.mean()
    std = signal.std()
    signal = (signal - mean) / std
    return signal


# Streamlit app
def main():
    # set to "soundfile" if on windows
    torchaudio.set_audio_backend("sox_io")
    st.title("Speaker Identification App")
    st.write("Upload an audio file and the model will predict the speaker.")

    uploaded_file = st.file_uploader(
        "Choose an audio file...", type=["wav", "mp3", "flac"]
    )

    speaker_map = create_speaker_map()
    if uploaded_file is not None:
        # Load the audio file

        with open("temp.flac", "wb") as f:
            f.write(uploaded_file.getbuffer())

        audio, sample_rate = torchaudio.backend.sox_io_backend.load("temp.flac")
        st.audio(uploaded_file, format="audio/wav")

        # Preprocess the audio
        SAMPLE_RATE = 16000
        NUM_SAMPLES = 16000 * 5
        transformation = nn.Sequential(
            MelSpectrogram(
                sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=128
            ),
            AmplitudeToDB(),
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        audio = load_and_preprocess(
            audio, sample_rate, transformation, NUM_SAMPLES, SAMPLE_RATE, device
        )

        # Load the model
        model = resnet18(10).to(device)
        model.load_state_dict(
            torch.load("voice_classification.pth", map_location=device)
        )

        # Make prediction
        prediction = predict(model, audio, device).item()
        st.write(f"Predicted Speaker: {speaker_map[prediction]}")


if __name__ == "__main__":
    main()
