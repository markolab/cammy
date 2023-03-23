// Note that this is heavily borrowed from https://github.com/ksseverson57/campy/blob/master/campy/trigger/trigger.ino

int dig_out_pins[20];								 /* array to store digital pin IDS */
int n_pins;											 /* number of digital pins */
unsigned long frame_start, frame_period, start_time; /* micros ticks when frame began and when frame ends */
float frame_rate = 0;								 /* frame rate specified by user */
int pulse_width = 500;								 /*Camera trigger pulse width*/
uint32_t baudrate = 115200;							 /* set by fiat */

void setup(void)
{
	Serial.begin(baudrate);
	delay(1);
	/* wait for a serial connection */
	while (!Serial)
		;

	Serial.print("Set baudrate to: ");
	Serial.print(baudrate);

	Serial.println("");
	Serial.println("Enter delimiter-separated string with: N PINS, PIN ID1, PIN IDN, FRAME_RATE");

	set_digital_pins();
	set_frame_rate();
	frame_period = frame_rate_to_period(frame_rate);
}

void set_digital_pins()
{
	// wait for input
	while (Serial.available() == 0)
	{
	}
	n_pins = int((unsigned int)Serial.parseFloat());

	Serial.println("");
	Serial.print("N(digital pins): ");
	Serial.print(n_pins);

	Serial.println("");
	Serial.print("Digital pins: ");

	for (int i = 0; i < n_pins; i++)
	{
		while (Serial.available() == 0)
		{
		}
		int pin_id = (unsigned int)Serial.parseFloat();
		Serial.print(pin_id);
		if (i + 1 < n_pins)
		{
			Serial.print(",");
		}
		dig_out_pins[i] = pin_id;
	}

	set_pins_low();
}

unsigned long frame_rate_to_period(unsigned long rate)
{
	if (rate == 0)
	{
		frame_period = 0xFFFFFFFF;
	}
	else
	{
		frame_period = 1e6 / rate; /* convert to micros ticks, period in secs is 1/rate */
	}

	Serial.println("");
	Serial.print("Set frame period to: ");
	Serial.print(frame_period);
	Serial.print(" microseconds");
	return frame_period;
}

void set_frame_rate()
{
	while (Serial.available() == 0)
	{
	}
	frame_rate = Serial.parseFloat();
	if (frame_rate < 0)
	{
		frame_rate = 0;
	}

	Serial.println("");
	Serial.print("Set frame rate to: ");
	Serial.print(frame_rate);
	Serial.print(" Hz");
}

void set_pins_low()
{
	noInterrupts();
	for (int i = 0; i < n_pins; i++)
	{
		digitalWrite(dig_out_pins[i], LOW);
	}
	interrupts();
}

void set_pins_high()
{
	noInterrupts();
	for (int i = 0; i < n_pins; i++)
	{
		digitalWrite(dig_out_pins[i], HIGH);
	}
	interrupts();
}

void loop(void)
{

	if (Serial.available())
	{
		set_pins_low();
		set_digital_pins();
		set_frame_rate();
		frame_period = frame_rate_to_period(frame_rate);
	}

	if (frame_rate > 0)
	{
		start_time = micros();

		set_pins_high(); /* pulse high at start */
		while (micros() - start_time < pulse_width)
		{
		}

		set_pins_low(); /* wait low until we're at the next period*/
		while (micros() - start_time < frame_period)
		{
		}
	}
}