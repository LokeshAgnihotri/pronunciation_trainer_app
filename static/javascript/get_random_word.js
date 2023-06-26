document.getElementById("next").addEventListener("click", next_word);
const next_button_audio = document.getElementById("next_button_audio");

let is_phonetic;
function next_word() {
    fetch('/next_word')
      .then(response => response.json()) // Parse the response as JSON
      .then(data => {
        const randomWord = data.random_word;
        const randomWordIpa = data.random_word_ipa;
        const randomWordAudio = data.pronunciation_audio;
        const  combined = randomWord + " " + randomWordIpa;

        document.getElementById("reftext").value = `${randomWord}`;

        document.getElementById("next_button_audio").src = randomWordAudio;
        is_phonetic = true;
      });
  }

next_button_audio.onclick = function() {
    next_button_audio.play();
}



