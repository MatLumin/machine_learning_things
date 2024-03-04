from typing import *;
import random;

#general functions 
def form_a_list_filled_with_zeros(size:int)->List[float|int]:
	output:List[float|int] = list();
	for i1 in range(size):
		output.append(0);
	return output;

def fill_list_with_random_numbers(list_pointer:List[float])->List[float]:
	output:List[float] = [];
	for i1 in range(list_pointer.__len__()):
		output.append(random.randint(0,10)*0.1);
	return output;

def do_dot_procduct(a:List[float|int], b:List[float|int])->float|int:
	output:float = 0;
	for i1 in range(len(b)):
		output += a[i1] * b[i1];
	return output;

def activation_function(input:float)->int:
	#output will be 1 or -1
	return int(input > 0);


class Data:
	def __init__(self, input_size, input_contnet, expected_output):
		self.input_size = input_size;
		self.input_contnet = input_contnet;
		self.expected_output = expected_output;

	def __str__(self)->str:
		return f"<Data: size:{self.input_size} contetn:{self.input_contnet} label:{self.expected_output}>";



class Precpetron:
	def __init__(self, input_size:int):
		self.input_size:Final[int] = input_size;
		self.weights = form_a_list_filled_with_zeros(input_size);
		self.weights = fill_list_with_random_numbers(self.weights);
		self.bias = 0.0;


	def train(self, data_set:List[Data], learning_rate:float, epoch_count:int)->None:
		epoches_done = 0;
		while epoch_count > epoches_done:
			for current_data_set_index, current_data in enumerate(data_set):

				print(f"EPOCH STARTED; {epoches_done}/{epoch_count}");

				current_input = current_data.input_contnet;
				expected_output = current_data.expected_output;
				predicition = do_dot_procduct(current_data.input_contnet, self.weights);
				predicition = activation_function(predicition);
				predicition += self.bias;
				for i0 in range(self.input_size):
					self.weights[i0] += learning_rate*predicition*current_input[i0];
					#w_i+=α∗y∗xi
				self.bias += learning_rate*expected_output;

				epoches_done += 1;
				print(self);


	def update(self):
		pass


	def test(self, data_set_for_testing:List[Data]):
		for current_data in data_set_for_testing:
			predicition = 0;
			for index in range(self.input_size):
				predicition += current_data.input_contnet[index]*self.weights[index];
			predicition += self.bias;	
			predicition = activation_function(predicition);
			print(
			f"""
			========
			predicted = {predicition}
			expected = {current_data.expected_output}
			"""
			);

		

	def __str__(self)->str:
		return f"""
		============
		PRECEPTRON VAR DUMP
		input_size = {self.input_size}
		weights = {self.weights}
		bias = {self.bias}
		============
		""";



if __name__ == "__main__":
	model = Precpetron(input_size=5);
	train_data_set:List[Data] = [
		Data(input_size=5, input_contnet=[1,1,1,1,1], expected_output=1),
		Data(input_size=5, input_contnet=[-1,-1,-1,-1,-1], expected_output=-1),
		Data(input_size=5, input_contnet=[1,-1,1,1,-1], expected_output=1),
		Data(input_size=5, input_contnet=[-1,1,-1,-1,1], expected_output=-1),
		];
	
	test_data_set:List[Data] = [
		Data(input_size=5, input_contnet=[1,1,1,1,1], expected_output=1),
		Data(input_size=5, input_contnet=[-1,-1,-1,-1,-1], expected_output=-1),
		Data(input_size=5, input_contnet=[1,-1,1,1,-1], expected_output=1),
		Data(input_size=5, input_contnet=[-1,-1,-1,-1,-1], expected_output=-1),
		];

	model.train(train_data_set, 0.02, 80);
	model.test(test_data_set);
	


