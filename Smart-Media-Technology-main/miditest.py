import tkinter as tk
import mido
import rtmidi

# Define Midi settings
MIDI = 'ON'  # Turn Midi Output 'ON' or 'OFF'
midi_channel = 1  # MIDI Output Channel
midi_control1 = 2  # First MIDI CC Message
midi_control2 = 3  # Second MIDI CC Message

# Define midi port
if MIDI == 'ON':
    port = mido.open_output('IAC-Driver Bus 2')

def send_midi_signal(control, value):
    if MIDI == 'ON':
        msg = mido.Message('control_change', channel=midi_channel, control=control, value=value)
        port.send(msg)
        print(f"Sent MIDI CC{control} with value {value}")

def send_cc1_signal():
    send_midi_signal(midi_control1, 127)

def send_cc2_signal():
    send_midi_signal(midi_control2, 127)

# Create the main window
root = tk.Tk()
root.title("MIDI Test Signals")

# Create buttons
button1 = tk.Button(root, text="Send CC1 Signal", command=send_cc1_signal)
button1.pack(pady=10)

button2 = tk.Button(root, text="Send CC2 Signal", command=send_cc2_signal)
button2.pack(pady=10)

# Start the GUI event loop
root.mainloop()
