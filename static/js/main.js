/* FlashRender — project page
   No dependencies. Scroll reveal, nav state, scene comparison, trajectory explorer,
   BibTeX copy. */
(function () {
  'use strict';

  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* -------------------------------------- synchronised playback helpers -- */

  function showStatus(el, on) {
    if (el) el.hidden = !on;
  }

  /* startTogether() is an async chain: waitAll -> whenPainted -> seekAllTo -> whenPainted.
     Scrolling fast (especially by dragging the scrollbar) can start the chain again for the
     same set of videos before the previous one finishes, or let a stale chain keep running
     after the set has scrolled out of view - two chains then take turns pausing, seeking and
     playing the same <video> elements, and some clips look frozen while others move.
     Each set (keyed on its first video) holds one token; a chain that is no longer current
     stops itself before moving to the next step. */
  var syncTokens = typeof WeakMap === 'function' ? new WeakMap() : null;

  function invalidateSync(videos, status) {
    if (!videos || !videos.length) return;
    if (syncTokens) syncTokens.set(videos[0], {});
    showStatus(status, false);
  }

  /* Wait until every video satisfies the condition.

     Listening for events alone misses them: assigning currentTime on an already-buffered
     video fires seeked before the listener is attached, and then the wait never ends.
     So events are only an accelerator here; the actual check is a short poll. */
  function waitAll(videos, isReady, events, timeoutMs, cb) {
    var settled = false;
    var remaining = videos.slice();
    var poll, timer;

    function cleanup() {
      clearInterval(poll);
      clearTimeout(timer);
      videos.forEach(function (v) {
        events.forEach(function (e) { v.removeEventListener(e, check); });
        v.removeEventListener('error', check);
      });
    }
    function finish() {
      if (settled) return;
      settled = true;
      cleanup();
      cb();
    }
    function check() {
      remaining = remaining.filter(function (v) { return !(isReady(v) || v.error); });
      if (!remaining.length) finish();
    }

    videos.forEach(function (v) {
      events.forEach(function (e) { v.addEventListener(e, check); });
      v.addEventListener('error', check);
      // With preload="none" nothing loads on its own, so kick it off - but only right after
      // src was set. readyState>=1 means it already loaded; calling load() again there resets
      // currentTime to 0 without going through the setter, which is exactly why the play
      // button used to jump back to the start every time.
      if (v.preload === 'none' && v.readyState === 0) v.load();
    });

    check();                        // it may already be satisfied
    if (settled) return;
    poll = setInterval(check, 50);
    timer = setTimeout(finish, timeoutMs);
  }

  /* Wait until each video has actually painted a frame to the screen.

     The media clock advancing (currentTime) and a frame being drawn are two different
     things. Start eight at once and the clocks line up within 0.01s, but decoders wake at
     different speeds and some paint their first frame hundreds of ms late. That reads as
     out of sync, and only looks right a few loops later once the decoders are warm.
     requestVideoFrameCallback fires when a frame is actually presented. */
  function whenPainted(videos, timeoutMs, cb) {
    var pending = videos.length;
    var settled = false;

    // cb also receives whether everything really painted - false when the timeout cut it
    // short. With more videos (a 3x4 grid, say) some decoders can be slow to wake, so the
    // caller gets to decide whether to warm up once more.
    function finish(allPainted) {
      if (settled) return;
      settled = true;
      cb(allPainted);
    }
    function tick() { if (--pending <= 0) finish(true); }

    videos.forEach(function (v) {
      if (typeof v.requestVideoFrameCallback === 'function') {
        v.requestVideoFrameCallback(function () { tick(); });
      } else {
        // Browsers without rVFC: fall back to a signal that playback has progressed.
        var on = function () { v.removeEventListener('timeupdate', on); tick(); };
        v.addEventListener('timeupdate', on);
      }
    });

    if (!videos.length) finish(true);
    setTimeout(function () { finish(false); }, timeoutMs);
  }

  function seekAllTo(videos, t, cb) {
    videos.forEach(function (v) {
      try { if (Math.abs(v.currentTime - t) > 0.001) v.currentTime = t; } catch (e) {}
    });
    waitAll(videos,
      function (v) { return !v.seeking && Math.abs(v.currentTime - t) < 0.02; },
      ['seeked'], 3000, cb);
  }

  function playAll(videos) {
    videos.forEach(function (v) {
      v.playbackRate = 1;
      var p = v.play();
      if (p && p.catch) p.catch(function () {});
    });
  }

  /* Once everything is ready, wake the decoders and start them together on the same frame.
     The overlay only lifts after every video has painted its first frame, so the ragged
     stretch before that is never shown. */
  function startTogether(videos, status, at) {
    if (!videos.length) return;
    // Without `at`, start from the beginning - the case when a new set of clips comes up.
    // With `at` (the play button on the frame slider, say) that time becomes the start line.
    var target = typeof at === 'number' ? at : 0;

    // Issue a token for this chain alone. If startTogether/invalidateSync runs again for
    // the same set, the token changes and every continuation below stops quietly.
    var key = videos[0];
    var myToken = {};
    if (syncTokens) syncTokens.set(key, myToken);
    function isCurrent() { return !syncTokens || syncTokens.get(key) === myToken; }

    showStatus(status, true);
    videos.forEach(function (v) { if (!v.paused) v.pause(); });

    // 1) Warm up: run once to wake the decoders and the compositor. With many videos
    //    (a 3x4 grid) one decoder may fail to wake within 4s, and moving on without it
    //    leaves the first few loops looking out of sync until it catches up. So warm-up
    //    repeats until everything has verifiably painted (at most 3 times).
    function prime(attemptsLeft) {
      playAll(videos);
      whenPainted(videos, 4000, function (allPainted) {
        if (!isCurrent()) return;
        videos.forEach(function (v) { v.pause(); });

        if (!allPainted && attemptsLeft > 0) {
          prime(attemptsLeft - 1);
          return;
        }

        // 2) line them up on the start frame, then 3) start them together again
        seekAllTo(videos, target, function () {
          if (!isCurrent()) return;
          playAll(videos);

          // 4) confirm every clip painted that frame, then lift the overlay.
          whenPainted(videos, 3000, function () {
            if (!isCurrent()) return;
            showStatus(status, false);
          });
        });
      });
    }

    waitAll(videos,
      function (v) { return v.readyState >= 4; },
      ['canplaythrough', 'loadeddata', 'canplay'],
      25000,
      function () {
        if (!isCurrent()) return;
        prime(2);
      });
  }

  /* Keep a set of videos aligned. Two things happen here.

     1) Rewinding is controlled here. Letting each clip loop on its own means each wraps at
        a different moment, so they drift apart every lap. When the first video ends, all of
        them are reset to 0 and started together.
     2) Small drift is corrected with playback rate, not seeking. Writing currentTime often
        forces re-buffering (readyState drops) and pulls them further apart instead. */
  function keepInSync(videos) {
    if (videos.length < 2) return;
    var master = videos[0];

    videos.forEach(function (v) { v.loop = false; });

    master.addEventListener('ended', function () {
      videos.forEach(function (v) {
        v.playbackRate = 1;
        try { v.currentTime = 0; } catch (e) {}
      });
      videos.forEach(function (v) {
        var p = v.play();
        if (p && p.catch) p.catch(function () {});
      });
    });

    setInterval(function () {
      if (master.paused || master.ended || master.readyState < 2) return;

      for (var i = 1; i < videos.length; i++) {
        var v = videos[i];
        if (v.seeking || v.ended || v.readyState < 2) continue;

        var delta = master.currentTime - v.currentTime;   // positive means this one lags
        var gap = Math.abs(delta);

        if (gap > 0.35) {
          // Too far gone for rate to catch up; cut once and realign.
          try { v.currentTime = master.currentTime; } catch (e) {}
          v.playbackRate = 1;
        } else if (gap > 0.02) {
          // At most +/-8%, closing the gap gradually and unnoticeably.
          v.playbackRate = Math.max(0.92, Math.min(1.08, 1 + delta * 0.6));
        } else if (v.playbackRate !== 1) {
          v.playbackRate = 1;
        }
      }
    }, 250);
  }

  /* ------------------------------------------------------------ reveal -- */

  var revealables = document.querySelectorAll('.reveal');

  if (!('IntersectionObserver' in window) || reduceMotion) {
    // Without the observer, simply leave everything visible.
    Array.prototype.forEach.call(revealables, function (el) { el.classList.add('is-in'); });
  } else {
    var revealObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        e.target.classList.add('is-in');
        revealObserver.unobserve(e.target);
      });
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.06 });

    Array.prototype.forEach.call(revealables, function (el) { revealObserver.observe(el); });
  }

  /* --------------------------------------------------------------- nav -- */

  var nav = document.querySelector('.nav');
  if (nav) {
    var onScroll = function () { nav.classList.toggle('is-stuck', window.scrollY > 8); };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  // Highlight the nav link for the section currently in view.
  var navLinks = document.querySelectorAll('.nav-links a[href^="#"]');
  if (navLinks.length && 'IntersectionObserver' in window) {
    var linkFor = {};
    var targets = [];

    Array.prototype.forEach.call(navLinks, function (a) {
      var id = a.getAttribute('href').slice(1);
      var section = document.getElementById(id);
      if (!section) return;
      linkFor[id] = a;
      targets.push(section);
    });

    var visible = {};
    var spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { visible[e.target.id] = e.isIntersecting; });

      // In document order, the first section overlapping the viewport counts as current.
      var currentId = null;
      for (var i = 0; i < targets.length; i++) {
        if (visible[targets[i].id]) { currentId = targets[i].id; break; }
      }
      Array.prototype.forEach.call(navLinks, function (a) {
        a.classList.toggle('is-current', currentId !== null && a === linkFor[currentId]);
      });
    }, { rootMargin: '-58px 0px -55% 0px' });

    targets.forEach(function (s) { spy.observe(s); });
  }

  /* Frame-lock slider.
     Comparing stills only works when every video in the grid holds the same frame.
     Dragging moves them all to that frame and pauses; play starts them together again. */
  var FPS = 30;   // every source clip is 30fps

  function buildScrubber(stage, vids, status) {
    var master = vids[0];
    var wrap = document.createElement('div');
    wrap.className = 'scrub';
    wrap.innerHTML =
      '<button class="scrub-toggle" type="button" aria-label="Pause"></button>' +
      '<input class="scrub-range" type="range" min="0" max="0" step="1" value="0" aria-label="Frame">' +
      '<span class="scrub-frame"></span>';
    stage.parentNode.insertBefore(wrap, stage.nextSibling);

    var toggle = wrap.querySelector('.scrub-toggle');
    var range = wrap.querySelector('.scrub-range');
    var label = wrap.querySelector('.scrub-frame');
    var frames = 0;

    function sizeUp() {
      if (!master.duration || !isFinite(master.duration)) return;
      frames = Math.max(1, Math.round(master.duration * FPS));
      range.max = frames - 1;
      paint(+range.value);
    }
    function paint(f) {
      label.textContent = 'frame ' + (f + 1) + ' / ' + frames;
      var pct = frames > 1 ? (f / (frames - 1)) * 100 : 0;
      range.style.setProperty('--pct', pct + '%');
    }
    function seekAll(f) {
      // Aim at the middle of the frame, or boundaries flicker between adjacent frames.
      var t = Math.min(master.duration - 0.001, (f + 0.5) / FPS);
      vids.forEach(function (v) { if (!v.paused) v.pause(); });
      // Just assigning currentTime on each and stopping there lets decoders seek at
      // different speeds, so some arrive first and others much later - which is what makes
      // them look independent while dragging. seekAllTo waits for every seeked event before
      // counting them as arrived.
      invalidateSync(vids, status);
      seekAllTo(vids, t, function () {});
    }
    function setPlaying(on) {
      stage.dataset.locked = on ? '' : '1';
      toggle.classList.toggle('is-playing', on);
      toggle.setAttribute('aria-label', on ? 'Pause' : 'Play');
    }

    master.addEventListener('loadedmetadata', sizeUp);
    if (master.readyState >= 1) sizeUp();

    range.addEventListener('input', function () {
      setPlaying(false);
      seekAll(+range.value);
      paint(+range.value);
    });

    toggle.addEventListener('click', function () {
      if (stage.dataset.locked === '1') {
        setPlaying(true);
        // Resume from where the slider parked, not from 0.
        var t = Math.min(master.duration - 0.001, (+range.value + 0.5) / FPS);
        startTogether(vids, status, t);
      } else {
        setPlaying(false);
        invalidateSync(vids, status);
        vids.forEach(function (v) { if (!v.paused) v.pause(); });
      }
    });

    // While playing, the slider tracks along.
    setInterval(function () {
      if (stage.dataset.locked === '1' || master.paused || !frames) return;
      var f = Math.min(frames - 1, Math.floor(master.currentTime * FPS));
      if (+range.value !== f) { range.value = f; paint(f); }
    }, 100);

    setPlaying(true);
    return { isLocked: function () { return stage.dataset.locked === '1'; } };
  }

  /* Fixed grids with no picker, like the distillation comparison: start them all at once
     when the grid scrolls into view. */
  Array.prototype.forEach.call(document.querySelectorAll('.stage'), function (stage) {
    var vids = Array.prototype.slice.call(stage.querySelectorAll('video[data-src]'));
    if (!vids.length || stage.closest('[data-scenes]')) return;
    var status = stage.querySelector('[data-status]');
    keepInSync(vids);
    var scrub = buildScrubber(stage, vids, status);

    if (!('IntersectionObserver' in window)) return;
    new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) {
          vids.forEach(function (v) {
            if (!v.getAttribute('src')) v.setAttribute('src', v.getAttribute('data-src'));
          });
          // If a frame is locked, do not resume playback on re-entry.
          if (scrub.isLocked()) return;
          if (vids.some(function (v) { return v.paused; })) startTogether(vids, status);
        } else {
          invalidateSync(vids, status);
          vids.forEach(function (v) { if (!v.paused) v.pause(); });
        }
      });
    }, { threshold: 0.2 }).observe(stage);
  });

  /* --------------------------------------------------- scene compare --- */

  Array.prototype.forEach.call(document.querySelectorAll('[data-scenes]'), function (root) {
    var tabs = root.querySelectorAll('.scene-tab[data-scene]');
    var panels = root.querySelectorAll('[data-scene-panel]');
    if (!tabs.length || !panels.length) return;

    var visible = false;

    function panelFor(slug) {
      for (var i = 0; i < panels.length; i++) {
        if (panels[i].getAttribute('data-scene-panel') === slug) return panels[i];
      }
      return null;
    }

    function videosOf(panel) {
      return Array.prototype.slice.call(panel.querySelectorAll('video'));
    }

    // Fill in the real src only when a scene is first selected. A scene is eight files, so
    // preloading them all would pull tens of MB on the very first screen.
    function load(panel) {
      videosOf(panel).forEach(function (v) {
        if (!v.getAttribute('src') && v.getAttribute('data-src')) {
          v.setAttribute('src', v.getAttribute('data-src'));
        }
      });
    }

    function pause(panel) {
      var vs = videosOf(panel);
      invalidateSync(vs, panel.querySelector('[data-status]'));
      vs.forEach(function (v) { if (!v.paused) v.pause(); });
    }

    var synced = [];   // attach the watcher once per panel
    function activate(panel) {
      load(panel);
      var vs = videosOf(panel);
      startTogether(vs, panel.querySelector('[data-status]'));
      if (synced.indexOf(panel) === -1) { synced.push(panel); keepInSync(vs); }
    }

    function select(tab) {
      var slug = tab.getAttribute('data-scene');
      Array.prototype.forEach.call(tabs, function (t) {
        var on = t === tab;
        t.classList.toggle('is-active', on);
        t.setAttribute('aria-selected', on ? 'true' : 'false');
      });
      Array.prototype.forEach.call(panels, function (pnl) {
        var on = pnl.getAttribute('data-scene-panel') === slug;
        pnl.classList.toggle('is-hidden', !on);
        if (!on) pause(pnl);
      });
      var panel = panelFor(slug);
      if (panel && visible) activate(panel);
    }

    Array.prototype.forEach.call(tabs, function (tab) {
      tab.addEventListener('click', function () { select(tab); });
    });

    var active = root.querySelector('.scene-tab.is-active') || tabs[0];
    Array.prototype.forEach.call(panels, function (pnl) {
      pnl.classList.toggle('is-hidden',
        pnl.getAttribute('data-scene-panel') !== active.getAttribute('data-scene'));
    });

    // Loading and playback start only on the first entry into view. Once a panel has
    // started it keeps playing even when scrolled away, so it is never rewound to 0 and
    // re-synced on every re-entry. (Switching scenes with a tab is separate - select()
    // still restarts them together from the beginning.)
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          visible = e.isIntersecting;
          if (!visible) return;
          var panel = root.querySelector('[data-scene-panel]:not(.is-hidden)');
          if (!panel || synced.indexOf(panel) !== -1) return;
          activate(panel);
        });
      }, { threshold: 0.15 }).observe(root);
    } else {
      visible = true;
      activate(panelFor(active.getAttribute('data-scene')));
    }
  });

  /* ------------------------------------------------ trajectory explorer -- */

  Array.prototype.forEach.call(document.querySelectorAll('[data-traj-explorer]'), function (root) {
    var trajBtns  = root.querySelectorAll('[data-traj]');
    var sceneBtns = root.querySelectorAll('[data-traj-scene]');
    var inputVid  = root.querySelector('video[data-role="input"]');
    var outVid    = root.querySelector('video[data-role="output"]');
    var trajName  = root.querySelector('[data-traj-name]');
    var status    = root.querySelector('[data-status]');
    if (!trajBtns.length || !sceneBtns.length || !inputVid || !outVid) return;

    var vids = [inputVid, outVid];
    var visible = false;
    keepInSync(vids);

    function activeOf(list, attr) {
      for (var i = 0; i < list.length; i++) {
        if (list[i].classList.contains('is-active')) return list[i];
      }
      return list[0];
    }

    function apply() {
      var scene = activeOf(sceneBtns, 'data-traj-scene').getAttribute('data-traj-scene');
      var trajBtn = activeOf(trajBtns, 'data-traj');
      var traj = trajBtn.getAttribute('data-traj');

      // The input does not depend on the trajectory; swap it only when the scene changes.
      var srcIn  = 'files/trajectories/' + scene + '/input.mp4';
      var srcOut = 'files/trajectories/' + scene + '/' + traj + '.mp4';
      var changed = false;
      if (inputVid.getAttribute('src') !== srcIn)  { inputVid.setAttribute('src', srcIn);  changed = true; }
      if (outVid.getAttribute('src')   !== srcOut) { outVid.setAttribute('src', srcOut);   changed = true; }

      if (trajName) trajName.textContent = trajBtn.textContent.trim();
      if (!visible) return;
      if (changed || inputVid.paused || outVid.paused) startTogether(vids, status);
    }

    function wire(list) {
      Array.prototype.forEach.call(list, function (btn) {
        btn.addEventListener('click', function () {
          Array.prototype.forEach.call(list, function (b) {
            var on = b === btn;
            b.classList.toggle('is-active', on);
            b.setAttribute('aria-selected', on ? 'true' : 'false');
          });
          apply();
        });
      });
    }
    wire(trajBtns);
    wire(sceneBtns);

    // Leaving the viewport does not pause anything; playback just continues. Only a
    // trajectory or scene button actually changing the content makes apply() re-sync.
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          visible = e.isIntersecting;
          if (visible) apply();
        });
      }, { threshold: 0.2 }).observe(root);
    } else {
      visible = true;
      apply();
    }
  });

  /* ------------------------------------------------------ bibtex copy --- */

  // navigator.clipboard exists only in a secure context (https or localhost).
  // Opened as http://<server-ip>:4000 it is undefined, so this falls back to execCommand.
  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise(function (resolve, reject) {
      var ta = document.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', '');
      ta.style.cssText = 'position:fixed;top:-9999px;opacity:0';
      document.body.appendChild(ta);
      ta.select();
      var ok = false;
      try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
      document.body.removeChild(ta);
      ok ? resolve() : reject();
    });
  }

  Array.prototype.forEach.call(document.querySelectorAll('[data-copy]'), function (btn) {
    btn.addEventListener('click', function () {
      var card = btn.closest('.bibtex-card');
      var code = card && card.querySelector('code');
      if (!code) return;

      function flash(label) {
        btn.textContent = label;
        btn.classList.add('is-done');
        setTimeout(function () {
          btn.textContent = 'Copy';
          btn.classList.remove('is-done');
        }, 1600);
      }

      copyText(code.textContent.trim())
        .then(function () { flash('Copied'); })
        .catch(function () { flash('Press ⌘/Ctrl+C'); });
    });
  });
})();
