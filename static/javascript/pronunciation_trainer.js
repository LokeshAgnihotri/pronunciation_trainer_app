const _button_pronounce_trainer = document.getElementById("button_pronounce_trainer");
const _reftext = document.getElementById("reftext");
const audio = document.getElementById("audio");

_button_pronounce_trainer.onclick = function () {
    next_button_audio.play();

    if (is_phonetic === false) {
        fetch("http://127.0.0.1:5000/pronunciation_trainer", {
            method: 'POST',
            headers: {
                'Content-type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(_reftext.value)
        })
            .then(response => {
                if (response.ok) {
                    return response.json();
                } else {
                    throw new Error("Something went wrong");
                }
            })
            .then(jsonResponse => {
                _reftext.value = jsonResponse['phenome'];
                is_phonetic=true;
                document.getElementById("audioPlayer").src = jsonResponse['sound'];
                audioPlayer.play(); // Play the audio
            })
            .catch(error => {
                alert(error.message);
            });
    }

}

//testing