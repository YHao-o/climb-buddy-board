// 约攀列表：固定 2s 刷新（多人访问不阻塞）
  document.addEventListener('DOMContentLoaded', () => {
    const REFRESH_MS = 2000
    const upcomingCountEl = document.getElementById('upcoming-count')
    const expiredCountEl = document.getElementById('expired-count')
    const upcomingGrid = document.getElementById('upcoming-grid')
    const expiredGrid = document.getElementById('expired-grid')
    const expiredDivider = document.getElementById('expired-divider')
    const emptyState = document.getElementById('empty-state')
    const modal = document.getElementById('confirm-modal')

    if (!upcomingGrid || !expiredGrid || !expiredDivider || !emptyState) return

    const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]))
    const escAttrWithNewlines = (s) => esc(s).replace(/\n/g, '&#10;')

    const buildInviteText = (e) => {
      const origin = window.location.origin
      const shareUrl = `${origin}/events/${e.id}`
      const host = e.host_nickname || e.nickname || ''
      const participants = Array.isArray(e.participants) ? e.participants : []
      const participantsText = participants.length > 0 ? ('- ' + participants.join('\\n- ')) : '暂无'
      return `🧗 约攀邀请\\n时间：${e.start_text_weekday || e.start_text || ''}` +
        `\\n地点：${e.location || ''}` +
        `\\n发起人：${host}` +
        `\\n已报名：${participants.length} 人` +
        `\\n报名昵称：\\n${participantsText}` +
        `\\n链接：${shareUrl}`
    }

    const eventSig = (e) => {
      const host = e.host_nickname || e.nickname || ''
      const participants = Array.isArray(e.participants) ? e.participants : []
      // ignore now_ts to prevent unnecessary re-render
      return [
        e.id,
        e.start_ts,
        e.start_text,
        e.start_text_weekday,
        e.location,
        host,
        participants.join('|'),
      ].join('~')
    }

    const listSig = (upcoming, expired) => {
      const up = (upcoming || []).map(eventSig).join('||')
      const ex = (expired || []).map(eventSig).join('||')
      return `${up}##${ex}`
    }

    const setTextIfChanged = (el, text) => {
      const t = String(text ?? '')
      if (el && el.textContent !== t) el.textContent = t
    }

    const ensureBlocksVisibility = (upcomingLen, expiredLen) => {
      expiredDivider.style.display = expiredLen > 0 ? '' : 'none'
      emptyState.style.display = (upcomingLen === 0 && expiredLen === 0) ? '' : 'none'
    }

    const setUpcomingParticipants = (card, e) => {
      const participants = Array.isArray(e.participants) ? e.participants : []
      const partSig = participants.join('\u0001')
      if (card.dataset.partSig === partSig) return
      card.dataset.partSig = partSig

      const participantsBox = card.querySelector('.participants')
      if (!participantsBox) return

      const titleEl = participantsBox.querySelector('.participants__title')
      if (titleEl) titleEl.textContent = `已报名 (${participants.length}人)`

      const oldList = participantsBox.querySelector('.participants__list')
      const oldEmpty = participantsBox.querySelector('.participants__empty')
      if (oldList) oldList.remove()
      if (oldEmpty) oldEmpty.remove()

      if (participants.length > 0) {
        const list = document.createElement('div')
        list.className = 'participants__list'
        participants.forEach(p => {
          const wrap = document.createElement('span')
          wrap.className = 'participant'
          wrap.appendChild(document.createTextNode(String(p)))

          const form = document.createElement('form')
          form.method = 'post'
          form.action = `/events/${e.id}/leave`
          form.className = 'inline'
          form.dataset.confirm = `确认取消报名？\n活动：${e.start_text}｜${e.location}\n昵称：${p}`

          const hidden = document.createElement('input')
          hidden.type = 'hidden'
          hidden.name = 'nickname'
          hidden.value = String(p)
          form.appendChild(hidden)

          const btn = document.createElement('button')
          btn.type = 'submit'
          btn.className = 'participant__remove'
          btn.title = '取消报名'
          btn.textContent = '×'
          form.appendChild(btn)

          wrap.appendChild(form)
          list.appendChild(wrap)
        })
        const joinForm = participantsBox.querySelector('form.participants__join')
        participantsBox.insertBefore(list, joinForm || null)
      } else {
        const empty = document.createElement('p')
        empty.className = 'participants__empty'
        empty.textContent = '暂无人报名，快来第一个！'
        const joinForm = participantsBox.querySelector('form.participants__join')
        participantsBox.insertBefore(empty, joinForm || null)
      }
    }

    const setExpiredParticipants = (card, e) => {
      const participants = Array.isArray(e.participants) ? e.participants : []
      const partSig = participants.join('\u0001')
      if (card.dataset.partSig === partSig) return
      card.dataset.partSig = partSig

      const participantsBox = card.querySelector('.participants')
      if (!participantsBox) return
      const titleEl = participantsBox.querySelector('.participants__title')
      if (titleEl) titleEl.textContent = `参与者 (${participants.length}人)`

      const oldList = participantsBox.querySelector('.participants__list')
      const oldEmpty = participantsBox.querySelector('.participants__empty')
      if (oldList) oldList.remove()
      if (oldEmpty) oldEmpty.remove()

      if (participants.length > 0) {
        const list = document.createElement('div')
        list.className = 'participants__list'
        participants.forEach(p => {
          const sp = document.createElement('span')
          sp.className = 'participant'
          sp.textContent = String(p)
          list.appendChild(sp)
        })
        participantsBox.appendChild(list)
      } else {
        const empty = document.createElement('p')
        empty.className = 'participants__empty'
        empty.textContent = '—'
        participantsBox.appendChild(empty)
      }
    }

    const updateUpcomingCardMeta = (card, e) => {
      const host = e.host_nickname || e.nickname || ''
      const metaSig = [e.start_ts, e.start_text, e.start_text_weekday, e.location, host].join('|')
      if (card.dataset.metaSig !== metaSig) {
        card.dataset.metaSig = metaSig
        const timeEl = card.querySelector('.event-card__time')
        const locEl = card.querySelector('.event-card__location')
        const hostEl = card.querySelector('.event-card__host strong')
        setTextIfChanged(timeEl, e.start_text_weekday || e.start_text || '')
        setTextIfChanged(locEl, e.location || '')
        setTextIfChanged(hostEl, host)

        const joinForm = card.querySelector('form.participants__join')
        if (joinForm) joinForm.action = `/events/${e.id}/join`
        const delForm = card.querySelector('form[action^="/events/"][action$="/delete"]')
        if (delForm) {
          delForm.action = `/events/${e.id}/delete`
          delForm.dataset.confirm = `确认删除这条活动？\n${e.start_text}｜${e.location}｜${host}`
        }
      }

      const inviteBtn = card.querySelector('[data-invite-text]')
      if (inviteBtn) inviteBtn.dataset.inviteText = buildInviteText(e)
    }

    const updateExpiredCardMeta = (card, e) => {
      const host = e.host_nickname || e.nickname || ''
      const metaSig = [e.start_ts, e.start_text, e.start_text_weekday, e.location, host].join('|')
      if (card.dataset.metaSig !== metaSig) {
        card.dataset.metaSig = metaSig
        const timeEl = card.querySelector('.event-card__time')
        const locEl = card.querySelector('.event-card__location')
        const hostEl = card.querySelector('.event-card__host strong')
        setTextIfChanged(timeEl, e.start_text_weekday || e.start_text || '')
        setTextIfChanged(locEl, e.location || '')
        setTextIfChanged(hostEl, host)

        const delForm = card.querySelector('form[action^="/events/"][action$="/delete"]')
        if (delForm) {
          delForm.action = `/events/${e.id}/delete`
          delForm.dataset.confirm = `确认删除这条活动？\n${e.start_text}｜${e.location}｜${host}`
        }
      }
    }

    const createUpcomingCard = (e) => {
      const host = e.host_nickname || e.nickname || ''
      const participants = Array.isArray(e.participants) ? e.participants : []
      const participantsHtml = participants.length > 0
        ? `<div class="participants__list">` + participants.map(p => {
          const confirmText = `确认取消报名？\n活动：${e.start_text}｜${e.location}\n昵称：${p}`
          return `<span class="participant">${esc(p)}` +
            `<form method="post" action="/events/${esc(e.id)}/leave" class="inline" data-confirm="${escAttrWithNewlines(confirmText)}">` +
            `<input type="hidden" name="nickname" value="${esc(p)}" />` +
            `<button type="submit" class="participant__remove" title="取消报名">×</button>` +
            `</form></span>`
        }).join('') + `</div>`
        : `<p class="participants__empty">暂无人报名，快来第一个！</p>`

      const inviteText = buildInviteText(e)
      const deleteConfirm = `确认删除这条活动？\n${e.start_text}｜${e.location}｜${host}`
      const html = `<article class="event-card" data-event-id="${esc(e.id)}">` +
        `<div class="event-card__header">` +
        `<div class="event-card__time">${esc(e.start_text_weekday || e.start_text || '')}</div>` +
        `<span class="event-card__status event-card__status--active">进行中</span>` +
        `</div>` +
        `<div class="event-card__body">` +
        `<div class="event-card__location">${esc(e.location || '')}</div>` +
        `<div class="event-card__host">发起人：<strong>${esc(host)}</strong></div>` +
        `</div>` +
        `<div class="participants">` +
        `<div class="participants__title">已报名 (${participants.length}人)</div>` +
        `${participantsHtml}` +
        `<form method="post" action="/events/${esc(e.id)}/join" class="participants__join">` +
        `<input class="input" name="nickname" placeholder="我也要去（填昵称）" required />` +
        `<button class="btn btn--primary btn--small" type="submit">🙋 报名</button>` +
        `</form>` +
        `</div>` +
        `<div class="event-card__actions">` +
        `<button class="btn btn--ghost btn--small" type="button" data-invite-text="${esc(inviteText)}">📣 邀请</button>` +
        `<form method="post" action="/events/${esc(e.id)}/delete" class="inline" data-confirm="${escAttrWithNewlines(deleteConfirm)}">` +
        `<button class="btn btn--ghost btn--small" type="submit">🗑️ 删除</button>` +
        `</form>` +
        `</div>` +
        `</article>`
      const tpl = document.createElement('template')
      tpl.innerHTML = html
      return tpl.content.firstElementChild
    }

    const createExpiredCard = (e) => {
      const host = e.host_nickname || e.nickname || ''
      const participants = Array.isArray(e.participants) ? e.participants : []
      const participantsHtml = participants.length > 0
        ? `<div class="participants__list">` + participants.map(p => `<span class="participant">${esc(p)}</span>`).join('') + `</div>`
        : `<p class="participants__empty">—</p>`
      const deleteConfirm = `确认删除这条活动？\n${e.start_text}｜${e.location}｜${host}`
      const html = `<article class="event-card event-card--expired" data-event-id="${esc(e.id)}">` +
        `<div class="event-card__header">` +
        `<div class="event-card__time">${esc(e.start_text_weekday || e.start_text || '')}</div>` +
        `<span class="event-card__status event-card__status--expired">已过期</span>` +
        `</div>` +
        `<div class="event-card__body">` +
        `<div class="event-card__location">${esc(e.location || '')}</div>` +
        `<div class="event-card__host">发起人：<strong>${esc(host)}</strong></div>` +
        `</div>` +
        `<div class="participants">` +
        `<div class="participants__title">参与者 (${participants.length}人)</div>` +
        `${participantsHtml}` +
        `</div>` +
        `<div class="event-card__actions">` +
        `<form method="post" action="/events/${esc(e.id)}/delete" class="inline" data-confirm="${escAttrWithNewlines(deleteConfirm)}">` +
        `<button class="btn btn--ghost btn--small" type="submit">🗑️ 删除</button>` +
        `</form>` +
        `</div>` +
        `</article>`
      const tpl = document.createElement('template')
      tpl.innerHTML = html
      return tpl.content.firstElementChild
    }

    const patchGrid = (grid, events, type) => {
      const existing = new Map()
      grid.querySelectorAll('article[data-event-id]').forEach(node => {
        existing.set(String(node.dataset.eventId), node)
      })

      const seen = new Set()
      let prev = null
      events.forEach(e => {
        const id = String(e.id)
        let node = existing.get(id)
        if (!node) {
          node = type === 'upcoming' ? createUpcomingCard(e) : createExpiredCard(e)
        }

        // Update only changed parts
        if (type === 'upcoming') {
          updateUpcomingCardMeta(node, e)
          setUpcomingParticipants(node, e)
        } else {
          updateExpiredCardMeta(node, e)
          setExpiredParticipants(node, e)
        }

        // Ensure order matches server (events already sorted)
        if (prev === null) {
          if (grid.firstElementChild !== node) {
            grid.insertBefore(node, grid.firstElementChild)
          }
        } else {
          if (prev.nextElementSibling !== node) {
            grid.insertBefore(node, prev.nextElementSibling)
          }
        }
        prev = node
        seen.add(id)
      })

      existing.forEach((node, id) => {
        if (!seen.has(id)) node.remove()
      })
    }

    let inFlight = false
    let controller = null
    let lastPayload = ''

    const shouldSkipRefresh = () => {
      const active = document.activeElement
      if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA' || active.tagName === 'SELECT')) {
        return true
      }
      if (modal && modal.classList.contains('modal--open')) return true
      return false
    }

    const tick = async () => {
      if (inFlight || shouldSkipRefresh()) return
      inFlight = true
      if (controller) controller.abort()
      controller = new AbortController()
      try {
        const res = await fetch(`/api/events?t=${Date.now()}`, {
          method: 'GET',
          cache: 'no-store',
          signal: controller.signal,
          headers: { 'Accept': 'application/json' }
        })
        if (!res.ok) return
        const data = await res.json()
        const upcoming = Array.isArray(data.upcoming) ? data.upcoming : []
        const expired = Array.isArray(data.expired) ? data.expired : []

        const sig = listSig(upcoming, expired)
        if (sig === lastPayload) return
        lastPayload = sig

        setTextIfChanged(upcomingCountEl, upcoming.length)
        setTextIfChanged(expiredCountEl, expired.length)
        ensureBlocksVisibility(upcoming.length, expired.length)

        patchGrid(upcomingGrid, upcoming, 'upcoming')
        patchGrid(expiredGrid, expired, 'expired')
      } catch (e) {
        // ignore
      } finally {
        inFlight = false
      }
    }

    // 立即刷新一次，然后固定 2s
    tick()
    window.setInterval(tick, REFRESH_MS)
  })



  // 地点：移动端友好的可输入 + 可点选
  document.addEventListener('DOMContentLoaded', () => {
    const combo = document.getElementById('location-combo')
    const input = document.getElementById('climbing-location')
    const panel = document.getElementById('location-suggest')
    const openBtn = document.getElementById('location-open')
    const datalist = document.getElementById('climbing-gym')
    if (!combo || !input || !panel || !openBtn || !datalist) return

    const all = Array.from(datalist.querySelectorAll('option'))
      .map(o => String(o.getAttribute('value') || '').trim())
      .filter(Boolean)
    const uniq = Array.from(new Set(all))

    const isMobile = () => window.matchMedia && window.matchMedia('(max-width: 600px)').matches

    const setExpanded = (v) => {
      input.setAttribute('aria-expanded', v ? 'true' : 'false')
    }

    const hidePanel = () => {
      if (panel.hidden) return
      panel.hidden = true
      panel.innerHTML = ''
      setExpanded(false)
    }

    const showPanel = (items) => {
      panel.innerHTML = ''
      if (!items || items.length === 0) {
        hidePanel()
        return
      }
      items.forEach(text => {
        const btn = document.createElement('button')
        btn.type = 'button'
        btn.className = 'combo__item'
        btn.textContent = text
        btn.addEventListener('click', () => {
          input.value = text
          hidePanel()
          input.focus()
        })
        panel.appendChild(btn)
      })
      panel.hidden = false
      setExpanded(true)
    }

    const filter = (q) => {
      const s = String(q || '').trim()
      if (!s) return uniq.slice(0, 10)
      const ss = s.toLowerCase()
      const starts = []
      const contains = []
      for (const v of uniq) {
        const vv = v.toLowerCase()
        if (vv.startsWith(ss)) starts.push(v)
        else if (vv.includes(ss)) contains.push(v)
        if (starts.length + contains.length >= 10) break
      }
      return starts.concat(contains).slice(0, 10)
    }

    const refresh = () => {
      if (isMobile()) return // 移动端用底部弹层，避免和键盘打架
      showPanel(filter(input.value))
    }

    input.addEventListener('focus', refresh)
    input.addEventListener('input', refresh)
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') hidePanel()
    })
    document.addEventListener('click', (e) => {
      const t = e.target
      if (!(t instanceof Node)) return
      if (!combo.contains(t)) hidePanel()
    })
    input.addEventListener('blur', () => {
      // 允许点击建议项
      window.setTimeout(hidePanel, 120)
    })

    const ensureModal = () => {
      let modal = document.getElementById('location-picker-modal')
      if (modal) return modal

      modal = document.createElement('div')
      modal.id = 'location-picker-modal'
      modal.className = 'modal modal--picker'
      modal.setAttribute('aria-hidden', 'true')
      modal.setAttribute('hidden', '')

      modal.innerHTML = `
        <div class="modal__backdrop" data-action="close"></div>
        <div class="modal__dialog modal__dialog--sheet" role="dialog" aria-modal="true" aria-labelledby="locpicker-title">
          <div class="picker">
            <div class="picker__header">
              <div class="picker__title" id="locpicker-title">选择地点</div>
              <button class="btn btn--ghost btn--small" type="button" data-action="close" aria-label="关闭">✕</button>
            </div>
            <div class="picker__search">
              <input class="input" type="search" placeholder="搜索地点（也可以直接手动输入）" />
            </div>
            <div class="picker__list" role="listbox" aria-label="地点列表"></div>
            <div class="picker__footer">
              <button class="btn btn--secondary" type="button" data-action="clear">清空</button>
              <button class="btn btn--primary" type="button" data-action="use">使用当前输入</button>
            </div>
          </div>
        </div>
      `
      document.body.appendChild(modal)
      return modal
    }

    const openModal = () => {
      const modal = ensureModal()
      const list = modal.querySelector('.picker__list')
      const search = modal.querySelector('.picker__search input')
      if (!(list instanceof HTMLElement) || !(search instanceof HTMLInputElement)) return

      const render = (q) => {
        const items = filter(q)
        list.innerHTML = ''
        items.forEach(v => {
          const b = document.createElement('button')
          b.type = 'button'
          b.className = 'picker__item'
          b.textContent = v
          b.addEventListener('click', () => {
            input.value = v
            closeModal()
            input.focus()
          })
          list.appendChild(b)
        })
      }

      const closeModal = () => {
        modal.classList.remove('modal--open')
        modal.setAttribute('hidden', '')
        modal.setAttribute('aria-hidden', 'true')
      }

      render(input.value)
      search.value = input.value || ''
      search.oninput = () => render(search.value)

      modal.addEventListener('click', (ev) => {
        const target = ev.target
        if (!(target instanceof HTMLElement)) return
        const action = target.dataset.action
        if (action === 'close') closeModal()
        if (action === 'clear') {
          input.value = ''
          search.value = ''
          render('')
        }
        if (action === 'use') {
          input.value = search.value.trim()
          closeModal()
          input.focus()
        }
      }, { once: true })

      modal.classList.add('modal--open')
      modal.removeAttribute('hidden')
      modal.setAttribute('aria-hidden', 'false')
      window.setTimeout(() => search.focus(), 50)
    }

    openBtn.addEventListener('click', () => {
      if (isMobile()) openModal()
      else {
        input.focus()
        refresh()
      }
    })
  })

  // 发起约攀：设置默认时间
  document.addEventListener('DOMContentLoaded', () => {
    const dateControl = document.querySelector('#climbing-time')
    const defaultDate = new Date()
    // 判断时区, 调整为东8区时间
    defaultDate.setMinutes(defaultDate.getMinutes() - defaultDate.getTimezoneOffset())
    // 按1刻钟取整
    defaultDate.setMinutes(Math.ceil(defaultDate.getMinutes() / 15) * 15)
    dateControl.value = defaultDate.toISOString().slice(0, 16)
  })

  // 村历：固定 2s 刷新
  document.addEventListener('DOMContentLoaded', () => {
    const REFRESH_MS = 2000
    const boardBody = document.querySelector('.week-board__body')
    const copyBtn = document.querySelector('.week-board__header [data-copy-text]')
    if (!boardBody) return

    const origin = window.location.origin
    const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]))

    let lastSig = ''
    let inFlight = false

    // 根据 week_board JSON 生成签名，变化时才重渲
    const boardSig = (wb) => JSON.stringify(wb)

    // 根据 week_board JSON 构造一键复制的纯文本
    const buildCopyText = (wb) => {
      let text = '本村周历'
      for (const day of wb) {
        text += '\n\n' + day.title
        if (day.items && day.items.length > 0) {
          for (const item of day.items) {
            text += '\n- 时间：' + item.time_text + ' 地点：' + item.location
            if (item.is_active) {
              text += ' 报名：' + origin + item.detail_url
            }
          }
        } else {
          text += '\n- 暂无安排'
        }
      }
      return text
    }

    // 重新渲染周历 HTML
    const renderBoard = (wb) => {
      let html = ''
      for (const day of wb) {
        const todayCls = day.is_today ? ' week-day--today' : ''
        html += `<div class="week-day${todayCls}">`
        html += `<div class="week-day__title">${esc(day.title)}</div>`
        if (day.items && day.items.length > 0) {
          html += '<div class="week-day__items">'
          for (const item of day.items) {
            html += '<div class="week-day__item">'
            html += `<span class="week-day__time">${esc(item.time_text)}</span>`
            html += `<span class="week-day__location">${esc(item.location)}</span>`
            if (item.is_active) {
              const detailUrl = origin + item.detail_url
              html += `<a class="link week-day__link" href="${esc(detailUrl)}">报名</a>`
            }
            html += '</div>'
          }
          html += '</div>'
        } else {
          html += '<div class="week-day__empty">暂无安排</div>'
        }
        html += '</div>'
      }
      boardBody.innerHTML = html
    }

    const tick = async () => {
      if (inFlight) return
      inFlight = true
      try {
        const resp = await fetch('/api/week-board')
        if (!resp.ok) return
        const data = await resp.json()
        const wb = data.week_board || []
        const sig = boardSig(wb)
        if (sig === lastSig) return
        lastSig = sig
        renderBoard(wb)
        if (copyBtn) {
          copyBtn.setAttribute('data-copy-text', buildCopyText(wb))
        }
      } catch (e) {
        // ignore
      } finally {
        inFlight = false
      }
    }

    tick()
    window.setInterval(tick, REFRESH_MS)
  })


// Feed 动态流：5s 轮询
document.addEventListener('DOMContentLoaded', () => {
  const list = document.getElementById('feed-list')
  if (!list) return
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]))
  let lastSig = ''
  const render = (items) => {
    if (!items.length) { list.innerHTML = '<p class="view-placeholder">还没有动态</p>'; return }
    list.innerHTML = items.map(it =>
      `<div class="feed__item feed__item--${esc(it.action)}">` +
      `<span class="feed__icon">${esc(it.icon)}</span>` +
      `<span class="feed__text"><strong>${esc(it.actor)}</strong> ${esc(it.verb)} ${esc(it.when)} ${esc(it.location)}</span>` +
      `<span class="feed__time">${esc(it.relative_time)}</span></div>`
    ).join('')
  }
  const tick = async () => {
    try {
      const r = await fetch('/api/feed', { cache: 'no-store' })
      if (!r.ok) return
      const data = await r.json()
      const items = Array.isArray(data.feed) ? data.feed : []
      const sig = JSON.stringify(items)
      if (sig === lastSig) return
      lastSig = sig
      render(items)
    } catch (e) { /* ignore */ }
  }
  tick()
  window.setInterval(tick, 5000)
})
