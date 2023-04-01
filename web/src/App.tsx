import { Component, createResource, createSignal } from 'solid-js';
import 'flowbite';

const App: Component = () => {
	const [email, setEmail] = createSignal('');
	const [prediction, setPrediction] = createSignal<number>();

	return (
		<form action="">
			<div class="container">
				<label
					for="email"
					class="block mb-2 text-sm font-medium text-gray-900 dark:text-white"
				>
					Your email
				</label>
				<textarea
					value={email()}
					onchange={(e) => {
						setEmail(e.target.value);
					}}
					id="email"
					rows="4"
					class="block p-2.5 w-full text-sm text-gray-900 bg-gray-50 rounded-lg border border-gray-300 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
					placeholder="Write your thoughts here..."
				></textarea>
				<div class="flex justify-between mt-4">
					<span class="text-sm text-gray-500 dark:text-gray-400">
						{prediction() === 1 ? 'Spam' : 'Not Spam'}
					</span>
				</div>
				<button
					onclick={(e) => {
						e.preventDefault();
						console.log(email());
						// send email to server on localhost:5000/predict endpoint and get prediction
						fetch('http://localhost:5000/predict', {
							method: 'POST',
							headers: {
								'Content-Type': 'application/json',
							},
							body: JSON.stringify({ message: email() }),
						})
							// set prediction to the response from the server
							// .then((response) => response.json())
							// .then((data) => {
							// 	setPrediction(data.prediction);
							// });
							.then((response) => {
								if (response.ok) {
									return response.json();
								}
								throw new Error('Network response was not ok.');
							})
							.then((data) => {
								console.log('prediction: ', data);

								setPrediction(data.prediction);
							});
					}}
					// onSubmit={(e) => {
					// 	e.preventDefault();
					// 	console.log(email());
					// }}
					type="submit"
					class="block w-full px-4 py-2 mt-4 text-sm font-medium text-white bg-blue-500 border border-transparent rounded-md shadow-sm hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
				>
					Submit
				</button>
			</div>
		</form>
	);
};

export default App;
