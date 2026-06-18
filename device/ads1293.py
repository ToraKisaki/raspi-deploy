"""
ADS1293 driver for Raspberry Pi 5 using software (bit-banged) SPI on the
upper GPIO header so it never collides with an LCD wired to pins 1..26.

Pi 5 note: RPi.GPIO does not work on the Pi 5. We use `lgpio`, which is the
supported low-level GPIO library on the Pi 5 / Bookworm.

    sudo apt install python3-lgpio    # or: pip install lgpio

The driver exposes:
    ADS1293(config).start()          -> begins conversions
    ads.read_sample()                -> returns list[float] of N channels (mV)
    ads.stream(callback)             -> blocking loop calling callback(sample)

If no hardware / lgpio is present it raises HardwareUnavailable so callers can
fall back to the simulator.
"""

import time

try:
    import lgpio
    _HAVE_LGPIO = True
except Exception:  # pragma: no cover - only on non-Pi machines
    _HAVE_LGPIO = False

from config import (
    PIN_SCLK, PIN_MOSI, PIN_MISO, PIN_CS, PIN_DRDY,
    SPI_HALF_PERIOD, N_INPUT_CHANNELS,
)


class HardwareUnavailable(RuntimeError):
    pass


# ADS1293 register map (subset we need)
REG_CONFIG = 0x00          # main config (start conversion bit)
REG_FLEX_CH1_CN = 0x01     # channel 1 input mux
REG_FLEX_CH2_CN = 0x02
REG_FLEX_CH3_CN = 0x03
REG_CMDET_EN = 0x0A
REG_RLD_CN = 0x0C
REG_OSC_CN = 0x12
REG_AFE_RES = 0x13
REG_AFE_SHDN_CN = 0x14
REG_R2_RATE = 0x21
REG_R3_RATE_CH1 = 0x22
REG_DRDYB_SRC = 0x27
REG_CH_CNFG = 0x2F         # ECG channels enabled in data ready
REG_DATA_CH1_ECG = 0x37    # 3 bytes per channel, big-endian, signed
REG_DATA_CH2_ECG = 0x3A
REG_DATA_CH3_ECG = 0x3D

READ = 0x80  # MSB set = read


class ADS1293:
    def __init__(self, vref_mv=2400.0):
        if not _HAVE_LGPIO:
            raise HardwareUnavailable("lgpio not available (not on a Pi?)")
        self.vref_mv = vref_mv
        self.n = N_INPUT_CHANNELS
        self._h = lgpio.gpiochip_open(0)
        # outputs
        for pin in (PIN_SCLK, PIN_MOSI, PIN_CS):
            lgpio.gpio_claim_output(self._h, pin, 0)
        lgpio.gpio_write(self._h, PIN_CS, 1)   # CS idle high
        # inputs
        lgpio.gpio_claim_input(self._h, PIN_MISO)
        lgpio.gpio_claim_input(self._h, PIN_DRDY)

    # ---------- low level bit-bang SPI (mode 0) ----------
    def _xfer_byte(self, out_byte):
        in_byte = 0
        for i in range(7, -1, -1):
            lgpio.gpio_write(self._h, PIN_MOSI, (out_byte >> i) & 1)
            time.sleep(SPI_HALF_PERIOD)
            lgpio.gpio_write(self._h, PIN_SCLK, 1)   # rising edge: sample
            time.sleep(SPI_HALF_PERIOD)
            bit = lgpio.gpio_read(self._h, PIN_MISO)
            in_byte = (in_byte << 1) | (bit & 1)
            lgpio.gpio_write(self._h, PIN_SCLK, 0)
        return in_byte

    def _cs(self, level):
        lgpio.gpio_write(self._h, PIN_CS, level)

    def write_reg(self, addr, value):
        self._cs(0)
        self._xfer_byte(addr & 0x7F)   # MSB clear = write
        self._xfer_byte(value & 0xFF)
        self._cs(1)

    def read_reg(self, addr, n=1):
        self._cs(0)
        self._xfer_byte((addr & 0x7F) | READ)
        out = [self._xfer_byte(0x00) for _ in range(n)]
        self._cs(1)
        return out if n > 1 else out[0]

    # ---------- configuration ----------
    def start(self):
        """Standard 3-lead config, 200 sps-ish, channel 1 = abdominal lead."""
        self.write_reg(REG_CONFIG, 0x00)        # stop while configuring
        # input mux: positive/negative inputs for each channel (IN1..IN6)
        self.write_reg(REG_FLEX_CH1_CN, 0x11)   # CH1: IN1(+) IN2(-)
        self.write_reg(REG_FLEX_CH2_CN, 0x19)   # CH2: IN3(+) IN4(-)
        self.write_reg(REG_FLEX_CH3_CN, 0x2E)   # CH3: IN5(+) IN6(-)
        self.write_reg(REG_CMDET_EN, 0x07)      # common-mode detect on used pins
        self.write_reg(REG_RLD_CN, 0x04)        # right-leg drive on IN4
        self.write_reg(REG_OSC_CN, 0x04)        # use internal oscillator
        self.write_reg(REG_AFE_RES, 0x36)       # high-res ADC, all channels
        self.write_reg(REG_AFE_SHDN_CN, 0x00)   # power up all blocks
        self.write_reg(REG_R2_RATE, 0x02)       # R2 decimation
        self.write_reg(REG_R3_RATE_CH1, 0x02)   # R3 decimation ch1
        self.write_reg(REG_DRDYB_SRC, 0x08)     # DRDYB driven by ECG data ready
        self.write_reg(REG_CH_CNFG, 0x70)       # enable ch1..3 in data stream
        time.sleep(0.05)
        self.write_reg(REG_CONFIG, 0x01)        # START conversions
        time.sleep(0.05)

    # ---------- data ----------
    @staticmethod
    def _to_signed24(b2, b1, b0):
        raw = (b2 << 16) | (b1 << 8) | b0
        if raw & 0x800000:
            raw -= 1 << 24
        return raw

    def _counts_to_mv(self, counts):
        # 24-bit signed full scale maps to +/- Vref. Convert to millivolts.
        return (counts / float(1 << 23)) * self.vref_mv

    def data_ready(self):
        # DRDY is active low
        return lgpio.gpio_read(self._h, PIN_DRDY) == 0

    def read_sample(self):
        """Read all channels once; returns list of mV floats."""
        regs = (REG_DATA_CH1_ECG, REG_DATA_CH2_ECG, REG_DATA_CH3_ECG)
        out = []
        for r in regs[: self.n]:
            b = self.read_reg(r, 3)
            out.append(self._counts_to_mv(self._to_signed24(b[0], b[1], b[2])))
        return out

    def stream(self, callback, poll_dt=0.0):
        """Blocking loop: wait for DRDY, read, call callback(list_of_mv)."""
        while True:
            if self.data_ready():
                callback(self.read_sample())
            elif poll_dt:
                time.sleep(poll_dt)

    def close(self):
        try:
            self.write_reg(REG_CONFIG, 0x00)
        finally:
            lgpio.gpiochip_close(self._h)
