const _button_pronounce_trainer = document.getElementById("button_pronounce_trainer")
const _reftext = document.getElementById("reftext")
const audio = document.getElementById("audio`")
_reftext.onchange = function () {
    is_ref_text = true;
}

_reftext.input = function () {
    is_ref_text = true;
}

_button_pronounce_trainer.onclick = function () {
    if (!is_ref_text) {
        next_button_audio.play()
    } else {
        fetch("/pronunciation_trainer",
            {
                method: 'POST',
                headers: {
                    'Content-type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(_reftext.value)
            })
            .then(response => {
                if (response.ok) {
                    return response.json()
                } else {
                    throw new Error("Something went wrong");
                }
            }).then(jsonResponse => {
                is_ref_text = false;
                _reftext.value = _reftext.value + '\n' + '\n' + jsonResponse['phenome']  ;
                document.getElementById("next_button_audio").src = jsonResponse['sound'];
                next_button_audio.play();  // Play the audio
            }
        ).catch((error) => {
            // alert(error.message)
        })
    }
}
