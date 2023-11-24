// Note that this is heavily borrowed from https://github.com/ksseverson57/campy/blob/master/campy/trigger/trigger.ino

int dig_out_pins[20];											 /* array to store digital pin IDS */
int n_pins;														 /* number of digital pins */
unsigned long frame_start, frame_period, start_time, max_pulses; /* micros ticks when frame began and when frame ends */
// float frame_rate = 0;											 /* frame rate specified by user */
// int pulse_width = 500;											 /*Camera trigger pulse width*/
unsigned long pulse_widths[20];
int pulse_width_low;
int n_pulse_widths;
// int inter_frame_interal = 2000;									 /* need this so next exposure is ready...*/
uint32_t baudrate = 115200;										 /* set by fiat */
unsigned long counter = 0;
unsigned int pulse_counter = 0;
int light_response_time = 200; /* response time of lights */
int additional_time = 0;
int previous_light_pin = 3;
int alternate_condition = 0;

void setup(void)
{

	/* set all pins to low while we're waiting for commands */

	for (int i = 4; i < 14; i++)
	{
		pinMode(i, OUTPUT);
		digitalWrite(i, LOW);
	}

	Serial.begin(baudrate);
	// delay(1);
	// wait for a serial connection
	while (!Serial)
		;

	Serial.print("Set baudrate to: ");
	Serial.print(baudrate);

	Serial.println("");
	Serial.println("Enter delimiter-separated string with: N PINS, PIN ID1, PIN IDN, NPULSES, ALTERNATE_CONDITION, PULSE_WIDTH_LOW");

	set_digital_pins();
	// set_frame_rate();
	// set_max_pulses();
	set_alternate_condition();
	set_light_pins();
	set_pulse_widths();
	// frame_period = frame_rate_to_period(frame_rate);

	while (Serial.available() > 0)
	{
		Serial.parseFloat();
	}
}

void set_alternate_condition()
{
	while (Serial.available() == 0)
	{
	}
	alternate_condition = Serial.parseFloat();
}

void set_light_pins()
{
	pinMode(2, OUTPUT);
	pinMode(3, OUTPUT);
	if (alternate_condition == 0)
	{
		digitalWrite(2, LOW);
		digitalWrite(3, HIGH);
	}
	else if (alternate_condition == 1)
	{
		digitalWrite(2, HIGH);
		digitalWrite(3, LOW);
	}
	else
	{
		digitalWrite(2, HIGH);
		digitalWrite(3, HIGH);
	}

	Serial.println("");
	Serial.print("Set alternate condition to (0, 1 constant; 2 alternate): ");
	Serial.print(alternate_condition);
}

void set_pulse_widths()
{
	while (Serial.available() == 0)
	{
	}
	n_pulse_widths = int((unsigned int)Serial.parseFloat());

	Serial.println("");
	Serial.print("N(pulse widths): ");
	Serial.print(n_pulse_widths);

	Serial.println("");
	Serial.print("Pulse widths: ");

	for (int i = 0; i < n_pulse_widths; i++)
	{
		while (Serial.available() == 0)
		{
		}
	  long pulse_width = (unsigned long)Serial.parseFloat();
		Serial.print(pulse_width);
		if (i + 1 < n_pulse_widths)
		{
			Serial.print(",");
		}
		pulse_widths[i] = pulse_width;
	}

	while (Serial.available() == 0)
	{
	}
	pulse_width_low = (unsigned long)Serial.parseFloat();

	Serial.println("");
	Serial.print("Pulse width low: ");
	Serial.print(pulse_width_low);

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
		pinMode(pin_id, OUTPUT);
		Serial.print(pin_id);
		if (i + 1 < n_pins)
		{
			Serial.print(",");
		}
		dig_out_pins[i] = pin_id;
	}

	set_pins_low();
}

// unsigned long frame_rate_to_period(unsigned long rate)
// {
// 	if (rate == 0)
// 	{
// 		frame_period = 1000;
// 	}
// 	else
// 	{
// 		frame_period = 1e6 / rate; /* convert to micros ticks, period in secs is 1/rate */
// 	}

// 	Serial.println("");
// 	Serial.print("Set frame period to: ");
// 	Serial.print(frame_period);
// 	Serial.print(" microseconds");
// 	return frame_period;
// }

// void set_frame_rate()
// {
// 	while (Serial.available() == 0)
// 	{
// 	}
// 	frame_rate = Serial.parseFloat();
// 	if (frame_rate < 0)
// 	{
// 		frame_rate = 0;
// 	}

// 	Serial.println("");
// 	Serial.print("Set frame rate to: ");
// 	Serial.print(frame_rate);
// 	Serial.print(" Hz");
// }

// void set_max_pulses()
// {
// 	while (Serial.available() == 0)
// 	{
// 	}
// 	max_pulses = (unsigned long)Serial.parseFloat();
// 	Serial.println("");
// 	Serial.print("Set max pulses to: ");
// 	Serial.print(max_pulses);
// }

void set_light_pins_low()
{
	noInterrupts();
	digitalWrite(2, LOW);
	digitalWrite(3, LOW);
	interrupts();
}

void set_light_pins_high()
{
	noInterrupts();
	digitalWrite(2, HIGH);
	digitalWrite(3, HIGH);
	interrupts();
}

void alternate_light_pins()
{
	noInterrupts();
	if (previous_light_pin == 3)
	{
		digitalWrite(3, HIGH);
		digitalWrite(2, LOW);
		previous_light_pin = 2;
	}
	else if (previous_light_pin == 2)
	{
		digitalWrite(3, LOW);
		digitalWrite(2, HIGH);
		previous_light_pin = 3;
	}
	interrupts();
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

// unsigned long counter = 1;
void loop(void)
{

	if (Serial.available())
	{
		set_pins_low();
		set_light_pins_high();
		set_digital_pins();
		// set_frame_rate();
		// set_max_pulses();
		set_alternate_condition();
		set_light_pins();
		set_pulse_widths();
		// frame_period = frame_rate_to_period(frame_rate);
		counter = 0;
		pulse_counter = 0;
		while (Serial.available() > 0)
		{
			Serial.parseFloat();
		}
	}

	if (pulse_widths[pulse_counter] > 0)
	{
		start_time = micros();

		if (alternate_condition == 2)
		{
			alternate_light_pins(); /* wait 20-50 usec for lights to turn on */
			while (micros() - start_time < light_response_time)
			{
			}
			additional_time = light_response_time;
		}
		else
		{
			additional_time = 0;
		}

		set_pins_high(); /* pulse high at start */
		// while (micros() - start_time < pulse_widths[pulse_counter] + additional_time)
		while (micros() - start_time < pulse_widths[pulse_counter])
		{
		}

		set_pins_low(); /* wait low until we're at the next period*/
		// while (micros() - start_time < frame_period)
		// while (micros() - start_time < pulse_widths[pulse_counter] + additional_time + inter_frame_interal)
		while (micros() - start_time < pulse_widths[pulse_counter] + pulse_width_low)	
		{
		}
		counter++;
		pulse_counter = (pulse_counter + 1) % n_pulse_widths;
	}
}