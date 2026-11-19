import gc
import logging
import unittest
from unittest.mock import Mock, patch
import weakref

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

    def test_pool_storage_is_copied_without_calling_the_bytes_accessor(self):
        for pixel_format, dtype in [
            (Aravis.PIXEL_FORMAT_MONO_8, np.uint8),
            (Aravis.PIXEL_FORMAT_MONO_12, np.uint16),
            (Aravis.PIXEL_FORMAT_MONO_16, np.uint16),
            (Aravis.PIXEL_FORMAT_COORD3D_C_16, np.uint16),
        ]:
            with self.subTest(pixel_format=pixel_format):
                expected = np.arange(12, dtype=dtype).reshape(3, 4)
                pixels = np.frombuffer(expected.tobytes(), dtype=np.uint8).copy()
                buffer = self.make_buffer(b"", pixel_format)
                self.camera._buffer_data = {buffer: pixels}
                frame = self.camera._array_from_buffer_address(buffer)
                buffer.get_data.assert_not_called()
                np.testing.assert_array_equal(frame, expected)
                self.assertTrue(frame.flags.c_contiguous)
                self.assertFalse(np.shares_memory(frame, pixels))
                pixels[:] = 255
                np.testing.assert_array_equal(frame, expected)

    def test_preallocated_native_buffer_reuse_preserves_previous_pixels(self):
        fake = Aravis.FakeCamera.new("pool-copy")
        self.camera._payload = fake.get_payload()
        self.camera._buffer_data = {}
        buffer = self.camera._allocate_buffer()
        fake.fill_buffer(buffer)
        expected = buffer.get_data()

        with patch.object(buffer, "get_data", side_effect=AssertionError("Unexpected bytes copy")):
            frame = self.camera._array_from_buffer_address(buffer)
        self.assertEqual(frame.tobytes(), expected)
        fake.fill_buffer(buffer)
        self.assertNotEqual(buffer.get_data(), expected)
        self.assertEqual(frame.tobytes(), expected)

    def test_pixel_storage_lives_until_native_buffer_destruction(self):
        fake = Aravis.FakeCamera.new("pool-owner")
        self.camera._payload = fake.get_payload()
        self.camera._buffer_data = {}
        buffer = self.camera._allocate_buffer()
        owner = weakref.ref(self.camera._buffer_data[buffer])
        del self.camera
        gc.collect()

        self.assertIsNotNone(owner())
        fake.fill_buffer(buffer)  # Native code can still safely write after camera disposal.
        self.assertEqual(buffer.get_status(), Aravis.BufferStatus.SUCCESS)
        self.assertEqual(owner().tobytes(), buffer.get_data())
        del buffer
        gc.collect()
        self.assertIsNone(owner())

    def test_native_stream_round_trip_retains_the_pool_mapping(self):
        device = Aravis.FakeDevice.new("pool-stream")
        self.camera.camera = Aravis.Camera.new_with_device(device)
        self.camera.buffer_size = 3
        self.camera.logger = logging.getLogger(__name__)
        self.camera.initialize_acquisition_stream()
        buffer = None
        try:
            self.camera.start_acquisition()
            buffer = self.camera.stream.timeout_pop_buffer(1000000)
            self.assertIsNotNone(buffer)
            self.assertEqual(buffer.get_status(), Aravis.BufferStatus.SUCCESS)
            self.assertIn(buffer, self.camera._buffer_data)
            expected = buffer.get_data()
            with patch.object(buffer, "get_data", side_effect=AssertionError("Unexpected bytes copy")):
                frame = self.camera._array_from_buffer_address(buffer)
            self.assertEqual(frame.tobytes(), expected)
        finally:
            if buffer is not None:
                self.camera.stream.push_buffer(buffer)
            self.camera.stop_acquisition()

    def test_trailing_chunk_bytes_are_excluded(self):
        pixels = np.arange(12, dtype=np.uint16)
        buffer = self.make_buffer(pixels.tobytes() + b"chunk", Aravis.PIXEL_FORMAT_MONO_16)
        frame = self.camera._array_from_buffer_address(buffer)
        self.assertEqual(frame.shape, (3, 4))
        self.assertEqual(frame.tobytes(), pixels.tobytes())

    def test_short_payload_is_rejected_before_reading_outside_the_image(self):
        with self.assertRaises(ValueError):
            self.camera._array_from_buffer_address(self.make_buffer(b"short"))

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


if __name__ == "__main__":
    unittest.main()
