  function showStars(value) {
      const starRating = document.getElementById("starRating");
      starRating.innerHTML = "";

      if (value === 100) {
        const burstAnimation = document.createElement("div");
        burstAnimation.classList.add("burst-animation");
        starRating.appendChild(burstAnimation);
        setTimeout(() => {
          burstAnimation.remove();
          displayStars(value);
        }, 2000);
      } else {
        displayStars(value);
      }
    }

    function displayStars(value) {
      const starsContainer = document.createElement("div");
      starsContainer.classList.add("stars");

      if (value === 0) {
        for (let i = 0; i < 5; i++) {
          const emptyStar = document.createElement("span");
          emptyStar.classList.add("star", "red");
          emptyStar.innerHTML = "☆";
          starsContainer.appendChild(emptyStar);
        }
      } else {
        const stars = document.createElement("span");
        stars.classList.add("star");
        if (value >= 90) {
          stars.classList.add("golden");
        } else if (value >= 70) {
          stars.classList.add("yellow");
        } else if (value >= 50) {
          stars.classList.add("orange");
        } else {
          stars.classList.add("red");
        }

        if (value >= 90) {
          stars.innerHTML = "★★★★★";
        } else if (value >= 70) {
          stars.innerHTML = "★★★★☆";
        } else if (value >= 50) {
          stars.innerHTML = "★★★☆☆";
        } else if (value >= 20) {
          stars.innerHTML = "★★☆☆☆";
        } else {
          stars.innerHTML = "★☆☆☆☆";
        }

        starsContainer.appendChild(stars);
      }

      starRating.appendChild(starsContainer);

      if (value === 100) {
        stars.classList.add("animation");
      }
    }
