import unittest
from unittest.mock import Mock

import click

from cammy.camera.aravis import AravisCamera
from cammy.cli import apply_transmission_stagger, simple_preview


def gige_camera(hz, maximum=1_000_000):
    cam = Mock(spec=AravisCamera)
    cam.device = Mock()
    cam.device.get_timestamp_tick_frequency.return_value = hz
    cam.camera = Mock()
    cam.camera.is_gv_device.return_value = True
    cam.camera.get_integer_bounds.return_value = (0, maximum)
    cam.camera.get_integer.return_value = 42
    def set_integer(name, value):
        assert name == "GevSCFTD"
        cam.camera.get_integer.return_value = value
    cam.camera.set_integer.side_effect = set_integer
    return cam


class TransmissionStaggerTests(unittest.TestCase):
    def test_order_tick_conversion_and_restore(self):
        usb = gige_camera(0)
        usb.camera.is_gv_device.return_value = False
        cameras = {"C": gige_camera(125_000_000), "B": gige_camera(1_000_000_000),
                   "A": gige_camera(1_000_000_000), "AA": usb}
        with simple_preview.make_context("run", ["--transmission-stagger-us", "100"]) as ctx:
            applied = apply_transmission_stagger(cameras, ctx.params["transmission_stagger_us"])
            self.assertEqual(applied, {"A": 0, "B": 100_000, "C": 25_000})
            for key, ticks in applied.items():
                self.assertEqual(cameras[key].camera.get_integer("GevSCFTD"), ticks)
        usb.camera.set_integer.assert_not_called()
        for cam in cameras.values():
            self.assertEqual(cam.camera.get_integer("GevSCFTD"), 42)

    def test_later_camera_rejection_restores_earlier_camera(self):
        cameras = {"A": gige_camera(1_000_000), "B": gige_camera(1_000_000, maximum=99)}
        with self.assertRaisesRegex(click.ClickException, "B:.*outside camera range"):
            with click.Context(simple_preview):
                apply_transmission_stagger(cameras, 100)
        self.assertEqual(cameras["A"].camera.get_integer("GevSCFTD"), 42)
        cameras["B"].camera.set_integer.assert_not_called()

    def test_explicit_zero_and_omitted_option(self):
        with simple_preview.make_context("run", []) as ctx:
            self.assertIsNone(ctx.params["transmission_stagger_us"])
        cameras = {key: gige_camera(1_000_000) for key in ("A", "B")}
        with simple_preview.make_context("run", ["--transmission-stagger-us", "0"]) as ctx:
            self.assertEqual(apply_transmission_stagger(cameras, ctx.params["transmission_stagger_us"]),
                             {"A": 0, "B": 0})
