document.getElementById("next").addEventListener("click", next_word);
const next_button_audio = document.getElementById("next_button_audio");


function next_word() {
    fetch('http://127.0.0.1:5000/next_word')
      .then(response => response.json()) // Parse the response as JSON
      .then(data => {
        const randomWord = data.random_word;
        const randomWordIpa = data.random_word_ipa;
        const randomWordAudio = data.pronunciation_audio;

        document.getElementById("reftext").value = `Your random word is: ${randomWord} and its phonetics are: ${randomWordIpa}`;

        document.getElementById("next_button_audio").src = randomWordAudio;
      });
  }

next_button_audio.onclick = function() {
    next_button_audio.play();
}



