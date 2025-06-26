document.addEventListener('DOMContentLoaded', function() {
    const packArea = document.getElementById('packArea');
    const loader = document.getElementById('loader');


    document.getElementById('generateDragonBtn').addEventListener('click',  async function() {
        // Get the values from your input fields
        const dragonName = document.getElementById('dragonName').value;
        const imagePrompt = document.getElementById('imagePrompt').value;
        
        // Your dragon generation logic here
        console.log('Dragon Name:', dragonName);
        console.log('Image Prompt:', imagePrompt);

        loader.style.display = 'block';
        packArea.innerHTML = '';

        try {
            // Play pack opening sound
            const openSound = new Audio('https://www.soundjay.com/buttons/sounds/button-09.mp3');
            openSound.volume = 0.7;
            openSound.play().catch(e => console.log('Sound autoplay prevented:', e));
            
            const response = await fetch(`/api/generate-dragon?dragon_name=${encodeURIComponent(dragonName)}&image_prompt=${encodeURIComponent(imagePrompt)}`);
            if (!response.ok) {
                throw new Error('Failed to generate Card');
            }
            
            const Response = await response.json();

            console.log(Response);

            const Card = Response.dragon;

            console.log(Card);
            
            // Add some dramatic delay
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            displayCard(Card);
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to generate card. Please try again.');
        } finally {
            // Hide loader
            loader.style.display = 'none';
        }
        
    });

    function displayCard(card) {
        const cardContainer = document.createElement('div');
        cardContainer.className = 'card-container';
        
        const cardInner = document.createElement('div');
        cardInner.className = 'card-inner';
        
        // Front of card (Yu-Gi-Oh card back)
        const cardFront = document.createElement('div');
        cardFront.className = 'card-front';
        
        // Back of card (actual card face)
        const cardBack = document.createElement('div');
        cardBack.className = `card-back`;
        
        let cardContent = `
            <div class="card-image" style="background-image: url('${card.image_url}')"></div>
            <div class="cardTitle">${card.name}</div>
            <div class="card-atk">${card.attack}</div>
            <div class="card-def">${card.defense}</div>
    
        `;
        
        cardContent += `<div class="card-description">${card.description}</div>`;
        
        cardBack.innerHTML = cardContent;
        
        cardInner.appendChild(cardFront);
        cardInner.appendChild(cardBack);
        cardContainer.appendChild(cardInner);
        
        // Add click event to flip card
        cardContainer.addEventListener('click', function() {
            this.classList.toggle('flipped');
            // Add a sound effect when flipping
            const flipSound = new Audio('https://www.soundjay.com/misc/sounds/page-flip-01a.mp3');
            flipSound.volume = 0.5;
            flipSound.play().catch(e => console.log('Sound autoplay prevented:', e));
        });
        
        // Add animation (no delay needed for single card)
        cardContainer.style.animationDelay = '0s';
        packArea.appendChild(cardContainer);
    }

    
    function displayPack(pack) {
        pack.cards.forEach((card, index) => {
            const cardContainer = document.createElement('div');
            cardContainer.className = 'card-container';
            
            const cardInner = document.createElement('div');
            cardInner.className = 'card-inner';
            
            // Front of card (Yu-Gi-Oh card back)
            const cardFront = document.createElement('div');
            cardFront.className = 'card-front';
            
            // Back of card (actual card face)
            const cardBack = document.createElement('div');
            cardBack.className = `card-back`;
            
            let cardContent = `
                <div class="card-image" style="background-image: url('${card.image_url}')"></div>
                <div class="cardTitle">${card.name}</div>
        
            `;
            
            
            cardContent += `<div class="card-description">${card.description}</div>`;
            
            cardBack.innerHTML = cardContent;
            
            cardInner.appendChild(cardFront);
            cardInner.appendChild(cardBack);
            cardContainer.appendChild(cardInner);
            
            // Add click event to flip card
            cardContainer.addEventListener('click', function() {
                this.classList.toggle('flipped');
                // Add a sound effect when flipping
                const flipSound = new Audio('https://www.soundjay.com/misc/sounds/page-flip-01a.mp3');
                flipSound.volume = 0.5;
                flipSound.play().catch(e => console.log('Sound autoplay prevented:', e));
            });
            
            // Add delay to each card for staggered appearance
            cardContainer.style.animationDelay = `${index * 0.1}s`;
            packArea.appendChild(cardContainer);
        });
    }
});