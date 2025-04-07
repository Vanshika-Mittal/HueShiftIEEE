// Synchronize video playback
document.addEventListener('DOMContentLoaded', function() {
    const syncVideos = document.querySelectorAll('.sync-video');
    const additionalVideos = document.querySelectorAll('.additional-video');
    const ganVideos = document.querySelectorAll('.gan-sample video');
    
    let masterVideo = null;
    let playPromise = null;
    
    // Create array from NodeList for easier manipulation
    const syncVideoArray = Array.from(syncVideos);
    const additionalVideoArray = Array.from(additionalVideos);
    const ganVideoArray = Array.from(ganVideos);
    
    // Set the first video as the master if available
    if (syncVideoArray.length > 0) {
        masterVideo = syncVideoArray[0];
    }
    
    // Function to synchronize all videos to the master
    function syncToMaster() {
        if (!masterVideo) return;
        
        const currentTime = masterVideo.currentTime;
        
        syncVideoArray.forEach(video => {
            if (video !== masterVideo && Math.abs(video.currentTime - currentTime) > 0.1) {
                video.currentTime = currentTime;
            }
        });
    }
    
    // Function to play all videos
    function playAllVideos(videoArr) {
        videoArr.forEach(video => {
            // Play returns a promise, store it to handle potential errors
            playPromise = video.play();
            
            if (playPromise !== undefined) {
                playPromise.catch(error => {
                    console.warn('Auto-play blocked:', error);
                    // Create a play button overlay if autoplay is blocked
                    createPlayButton();
                });
            }
        });
    }
    
    // Function to pause all videos
    function pauseAllVideos(videoArr) {
        videoArr.forEach(video => {
            video.pause();
        });
    }
    
    // Create a play button overlay if autoplay is blocked
    function createPlayButton() {
        // Check if button already exists
        if (document.querySelector('.play-all-button')) return;
        
        const button = document.createElement('button');
        button.innerText = 'Play All Videos';
        button.className = 'play-all-button btn btn-primary';
        button.style.position = 'fixed';
        button.style.top = '20px';
        button.style.right = '20px';
        button.style.zIndex = '1000';
        
        button.addEventListener('click', function() {
            playAllVideos(syncVideoArray);
            playAllVideos(additionalVideoArray);
            playAllVideos(ganVideoArray);
            this.remove();
        });
        
        document.body.appendChild(button);
    }
    
    // Add event listeners to master video
    if (masterVideo) {
        // Synchronize on time update
        masterVideo.addEventListener('timeupdate', syncToMaster);
        
        // When master plays, play all videos
        masterVideo.addEventListener('play', () => playAllVideos(syncVideoArray));
        
        // When master pauses, pause all videos
        masterVideo.addEventListener('pause', () => pauseAllVideos(syncVideoArray));
        
        // When master ends, ensure all videos restart together
        masterVideo.addEventListener('ended', function() {
            syncVideoArray.forEach(video => {
                video.currentTime = 0;
            });
            playAllVideos(syncVideoArray);
        });
        
        // Add click handlers to all videos
        syncVideoArray.forEach(video => {
            video.addEventListener('click', function() {
                if (masterVideo.paused) {
                    playAllVideos(syncVideoArray);
                } else {
                    pauseAllVideos(syncVideoArray);
                }
            });
        });
        
        // Try to start all videos immediately
        window.addEventListener('load', function() {
            setTimeout(() => playAllVideos(syncVideoArray), 500);
        });
    }
    
    // Handle accordion videos
    const accordionButtons = document.querySelectorAll('.accordion-button');
    
    accordionButtons.forEach(button => {
        button.addEventListener('click', function() {
            const isExpanded = this.getAttribute('aria-expanded') === 'true';
            
            if (!isExpanded) {
                // Get the collapse element ID
                const collapseId = this.getAttribute('data-bs-target').substring(1);
                const collapse = document.getElementById(collapseId);
                
                // Wait for accordion animation to complete
                setTimeout(() => {
                    // Find videos in this accordion section
                    const accordionVideos = Array.from(collapse.querySelectorAll('.additional-video'));
                    
                    // Set attributes and play videos
                    accordionVideos.forEach(video => {
                        video.loop = true;
                        video.muted = true;
                        
                        // Add click handler for play/pause toggle
                        video.onclick = function() {
                            const videos = Array.from(collapse.querySelectorAll('.additional-video'));
                            if (this.paused) {
                                playAllVideos(videos);
                            } else {
                                pauseAllVideos(videos);
                            }
                        };
                    });
                    
                    // Play all videos in this accordion
                    playAllVideos(accordionVideos);
                }, 350); // Wait for accordion animation
            }
        });
    });
    
    // Handle GAN videos
    ganVideoArray.forEach(video => {
        // Set attributes
        video.loop = true;
        video.muted = true;
        
        // Add click handler
        video.addEventListener('click', function() {
            if (this.paused) {
                this.play().catch(e => console.warn('Could not play GAN video:', e));
            } else {
                this.pause();
            }
        });
    });
    
    // Auto-play GAN videos on page load
    window.addEventListener('load', function() {
        setTimeout(() => playAllVideos(ganVideoArray), 700);
    });
    
    // GAN filtering
    const ganFilters = document.querySelectorAll('.gan-filter');
    const ganSamples = document.querySelectorAll('.gan-sample');
    
    ganFilters.forEach(filter => {
        filter.addEventListener('click', () => {
            // Update active button
            ganFilters.forEach(f => f.classList.remove('active'));
            filter.classList.add('active');
            
            const complexity = filter.getAttribute('data-complexity');
            
            ganSamples.forEach(sample => {
                if (complexity === 'all' || sample.getAttribute('data-complexity') === complexity) {
                    sample.style.display = 'block';
                } else {
                    sample.style.display = 'none';
                }
            });
        });
    });
    
    // Add keyboard controls
    document.addEventListener('keydown', function(e) {
        if (!masterVideo) return;
        
        // Space bar toggles play/pause
        if (e.code === 'Space') {
            e.preventDefault();
            if (masterVideo.paused) {
                playAllVideos(syncVideoArray);
            } else {
                pauseAllVideos(syncVideoArray);
            }
        }
        
        // Right arrow advances videos by 1 second
        if (e.code === 'ArrowRight') {
            e.preventDefault();
            const newTime = Math.min(masterVideo.duration, masterVideo.currentTime + 1);
            syncVideoArray.forEach(video => {
                video.currentTime = newTime;
            });
        }
        
        // Left arrow rewinds videos by 1 second
        if (e.code === 'ArrowLeft') {
            e.preventDefault();
            const newTime = Math.max(0, masterVideo.currentTime - 1);
            syncVideoArray.forEach(video => {
                video.currentTime = newTime;
            });
        }
        
        // R key restarts all videos
        if (e.code === 'KeyR') {
            e.preventDefault();
            syncVideoArray.forEach(video => {
                video.currentTime = 0;
            });
            playAllVideos(syncVideoArray);
        }
    });
}); 