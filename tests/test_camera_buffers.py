import gc
import unittest
from unittest.mock import Mock

import numpy as np

from cammy.camera.aravis import Aravis, AravisCamera


class CameraBufferTests(unittest.TestCase):
    def setUp(self):
        self.camera = AravisCamera.__new__(AravisCamera)

    def make_buffer(self, data, pixel_format=Aravis.PIXEL_FORMAT_MONO_8, height=3, width=4):
        buffer = Mock()
        buffer.get_data.return_value = data
        buffer.get_image_pixel_format.return_value = pixel_format
        buffer.get_image_height.return_value = height
        buffer.get_image_width.return_value = width
        buffer.get_image_padding.return_value = (0, 0)
        return buffer

    def test_supported_formats_use_the_bytes_snapshot_without_a_second_copy(self):
        for pixel_format, dtype in [
            (Aravis.PIXEL_FORMAT_MONO_8, np.uint8),
            (Aravis.PIXEL_FORMAT_MONO_12, np.uint16),
            (Aravis.PIXEL_FORMAT_MONO_16, np.uint16),
            (Aravis.PIXEL_FORMAT_COORD3D_C_16, np.uint16),
        ]:
            with self.subTest(pixel_format=pixel_format):
                expected = np.arange(12, dtype=dtype).reshape(3, 4)
                data = expected.tobytes()
                frame = self.camera._array_from_buffer_address(self.make_buffer(data, pixel_format))
                np.testing.assert_array_equal(frame, expected)
                self.assertEqual(frame.dtype, dtype)
                self.assertTrue(frame.flags.c_contiguous)
                self.assertFalse(frame.flags.writeable)
                self.assertTrue(np.shares_memory(frame, np.frombuffer(data, dtype=dtype)))

    def test_mutable_data_and_views_are_copied_before_buffer_reuse(self):
        for kind in ("bytearray", "memoryview", "readonly_view"):
            with self.subTest(kind=kind):
                source = bytearray(range(12))
                data = source if kind == "bytearray" else memoryview(source)
                if kind == "readonly_view":
                    data = data.toreadonly()
                frame = self.camera._array_from_buffer_address(self.make_buffer(data))
                source[:] = b"\xff" * 12
                self.assertEqual(frame.tobytes(), bytes(range(12)))

    def test_trailing_chunk_bytes_are_excluded(self):
        pixels = np.arange(12, dtype=np.uint16)
        buffer = self.make_buffer(pixels.tobytes() + b"chunk", Aravis.PIXEL_FORMAT_MONO_16)
        frame = self.camera._array_from_buffer_address(buffer)
        self.assertEqual(frame.shape, (3, 4))
        self.assertEqual(frame.tobytes(), pixels.tobytes())

    def test_short_payload_is_rejected_before_reading_outside_the_image(self):
        with self.assertRaises(ValueError):
            self.camera._array_from_buffer_address(self.make_buffer(b"short"))

    def test_cropped_rows_exclude_transport_padding(self):
        buffer = self.make_buffer(b"abc.def.", height=2, width=3)
        buffer.get_image_padding.return_value = (1, 0)
        frame = self.camera._array_from_buffer_address(buffer)
        self.assertEqual(frame.tobytes(), b"abcdef")
        self.assertTrue(frame.flags.c_contiguous)

    def test_unsupported_format_is_rejected_before_reading_pixels(self):
        buffer = self.make_buffer(bytes(12), pixel_format=0)
        with self.assertRaisesRegex(RuntimeError, "No unpacking strategy"):
            self.camera._array_from_buffer_address(buffer)
        buffer.get_data.assert_not_called()

    def test_real_aravis_buffer_reuse_and_destruction_preserve_previous_pixels(self):
        fake = Aravis.FakeCamera.new("test-buf")
        buffer = Aravis.Buffer.new_allocate(fake.get_payload())
        fake.fill_buffer(buffer)
        self.assertEqual(buffer.get_status(), Aravis.BufferStatus.SUCCESS)
        frame = self.camera._array_from_buffer_address(buffer)
        original = frame.tobytes()

        fake.fill_buffer(buffer)
        replacement = self.camera._array_from_buffer_address(buffer)
        self.assertNotEqual(replacement.tobytes(), original)
        self.assertEqual(frame.tobytes(), original)
        del replacement, buffer, fake
        gc.collect()
        self.assertEqual(frame.tobytes(), original)

    def test_existing_stream_receives_cropped_and_full_frames(self):
        cam = self.camera
        cam.camera = Aravis.Camera.new_with_device(Aravis.FakeDevice.new("live-roi"))
        cam.buffer_size = 4
        cam._last_framegrab = [np.nan]
        cam._spoof_cameras = []
        cam.initialize_acquisition_stream()
        stream = cam.stream
        original = tuple(cam.camera.get_region())
        for region in [(48, 28, 128, 64), original]:
            cam.change_region(region)
            cam.start_acquisition()
            try:
                buffer = stream.timeout_pop_buffer(1_000_000)
                self.assertEqual(buffer.get_status(), Aravis.BufferStatus.SUCCESS)
                frame = cam._array_from_buffer_address(buffer)
                self.assertEqual(frame.shape, (region[3], region[2]))
                self.assertEqual(frame.nbytes, cam._payload)
                stream.push_buffer(buffer)
            finally:
                cam.stop_acquisition()
            self.assertIs(cam.stream, stream)


if __name__ == "__main__":
    unittest.main()
