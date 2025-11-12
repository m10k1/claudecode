// ゲームの設定
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');

// ゲーム状態
let gameRunning = false;
let score = 0;
let lives = 3;
let gameLoop;

// プレイヤー
const player = {
    x: canvas.width / 2 - 25,
    y: canvas.height - 60,
    width: 50,
    height: 40,
    speed: 5,
    bullets: []
};

// インベーダー
let invaders = [];
const invaderRows = 4;
const invaderCols = 8;
let invaderDirection = 1;
let invaderSpeed = 1;
let invaderDropDistance = 20;

// キー入力
const keys = {
    left: false,
    right: false,
    space: false
};

// イベントリスナー
document.getElementById('startBtn').addEventListener('click', startGame);
document.getElementById('restartBtn').addEventListener('click', restartGame);

document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') keys.left = true;
    if (e.key === 'ArrowRight') keys.right = true;
    if (e.key === ' ') {
        e.preventDefault();
        keys.space = true;
        shootBullet();
    }
});

document.addEventListener('keyup', (e) => {
    if (e.key === 'ArrowLeft') keys.left = false;
    if (e.key === 'ArrowRight') keys.right = false;
    if (e.key === ' ') keys.space = false;
});

// ゲーム開始
function startGame() {
    document.getElementById('startBtn').style.display = 'none';
    document.getElementById('restartBtn').style.display = 'none';
    document.getElementById('gameOver').style.display = 'none';

    score = 0;
    lives = 3;
    player.x = canvas.width / 2 - 25;
    player.bullets = [];
    invaderSpeed = 1;

    createInvaders();
    updateUI();

    gameRunning = true;
    gameLoop = setInterval(update, 1000 / 60); // 60 FPS
}

// ゲーム再スタート
function restartGame() {
    startGame();
}

// インベーダーの作成
function createInvaders() {
    invaders = [];
    invaderDirection = 1;

    for (let row = 0; row < invaderRows; row++) {
        for (let col = 0; col < invaderCols; col++) {
            invaders.push({
                x: col * 80 + 80,
                y: row * 60 + 50,
                width: 40,
                height: 30,
                alive: true
            });
        }
    }
}

// 弾を撃つ
function shootBullet() {
    if (!gameRunning) return;

    player.bullets.push({
        x: player.x + player.width / 2 - 2,
        y: player.y,
        width: 4,
        height: 10,
        speed: 7
    });
}

// ゲームループ
function update() {
    // クリア
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // プレイヤーの移動
    if (keys.left && player.x > 0) {
        player.x -= player.speed;
    }
    if (keys.right && player.x < canvas.width - player.width) {
        player.x += player.speed;
    }

    // プレイヤーの描画
    drawPlayer();

    // 弾の更新と描画
    updateBullets();

    // インベーダーの更新と描画
    updateInvaders();

    // 衝突判定
    checkCollisions();

    // ゲームオーバーチェック
    if (lives <= 0) {
        endGame('ゲームオーバー');
    }

    // クリアチェック
    if (invaders.every(inv => !inv.alive)) {
        invaderSpeed += 0.5;
        createInvaders();
        score += 100;
        updateUI();
    }
}

// プレイヤーの描画
function drawPlayer() {
    ctx.fillStyle = '#00ff88';

    // 宇宙船の本体
    ctx.beginPath();
    ctx.moveTo(player.x + player.width / 2, player.y);
    ctx.lineTo(player.x, player.y + player.height);
    ctx.lineTo(player.x + player.width, player.y + player.height);
    ctx.closePath();
    ctx.fill();

    // コックピット
    ctx.fillStyle = '#00ccff';
    ctx.beginPath();
    ctx.arc(player.x + player.width / 2, player.y + 15, 8, 0, Math.PI * 2);
    ctx.fill();
}

// 弾の更新
function updateBullets() {
    for (let i = player.bullets.length - 1; i >= 0; i--) {
        const bullet = player.bullets[i];
        bullet.y -= bullet.speed;

        // 画面外の弾を削除
        if (bullet.y < 0) {
            player.bullets.splice(i, 1);
            continue;
        }

        // 弾の描画
        ctx.fillStyle = '#ffff00';
        ctx.fillRect(bullet.x, bullet.y, bullet.width, bullet.height);
    }
}

// インベーダーの更新
function updateInvaders() {
    let shouldDrop = false;

    // 端に到達したかチェック
    for (let invader of invaders) {
        if (!invader.alive) continue;

        if ((invaderDirection === 1 && invader.x + invader.width >= canvas.width) ||
            (invaderDirection === -1 && invader.x <= 0)) {
            shouldDrop = true;
            break;
        }
    }

    // 方向転換と下降
    if (shouldDrop) {
        invaderDirection *= -1;
        for (let invader of invaders) {
            if (invader.alive) {
                invader.y += invaderDropDistance;

                // インベーダーがプレイヤーに到達
                if (invader.y + invader.height >= player.y) {
                    endGame('インベーダーが地球に到達！');
                    return;
                }
            }
        }
    }

    // インベーダーの移動と描画
    for (let invader of invaders) {
        if (!invader.alive) continue;

        invader.x += invaderDirection * invaderSpeed;

        // インベーダーの描画
        ctx.fillStyle = '#ff0044';

        // インベーダーのデザイン
        ctx.fillRect(invader.x + 5, invader.y, 30, 5);
        ctx.fillRect(invader.x, invader.y + 5, 40, 15);
        ctx.fillRect(invader.x + 5, invader.y + 20, 10, 10);
        ctx.fillRect(invader.x + 25, invader.y + 20, 10, 10);

        // 目
        ctx.fillStyle = '#ffff00';
        ctx.fillRect(invader.x + 10, invader.y + 8, 5, 5);
        ctx.fillRect(invader.x + 25, invader.y + 8, 5, 5);
    }
}

// 衝突判定
function checkCollisions() {
    for (let i = player.bullets.length - 1; i >= 0; i--) {
        const bullet = player.bullets[i];

        for (let j = 0; j < invaders.length; j++) {
            const invader = invaders[j];

            if (!invader.alive) continue;

            // 矩形の衝突判定
            if (bullet.x < invader.x + invader.width &&
                bullet.x + bullet.width > invader.x &&
                bullet.y < invader.y + invader.height &&
                bullet.y + bullet.height > invader.y) {

                // インベーダーを破壊
                invader.alive = false;
                player.bullets.splice(i, 1);
                score += 10;
                updateUI();

                // エフェクト
                drawExplosion(invader.x + invader.width / 2, invader.y + invader.height / 2);
                break;
            }
        }
    }
}

// 爆発エフェクト
function drawExplosion(x, y) {
    ctx.fillStyle = '#ffff00';
    ctx.beginPath();
    ctx.arc(x, y, 15, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#ff8800';
    ctx.beginPath();
    ctx.arc(x, y, 10, 0, Math.PI * 2);
    ctx.fill();
}

// UI更新
function updateUI() {
    document.getElementById('score').textContent = score;
    document.getElementById('lives').textContent = lives;
}

// ゲーム終了
function endGame(message) {
    gameRunning = false;
    clearInterval(gameLoop);

    document.getElementById('finalScore').textContent = score;
    document.getElementById('gameOver').querySelector('h2').textContent = message;
    document.getElementById('gameOver').style.display = 'block';
    document.getElementById('restartBtn').style.display = 'inline-block';
}

// 初期画面の描画
ctx.fillStyle = '#000';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.fillStyle = '#00ff88';
ctx.font = '30px Arial';
ctx.textAlign = 'center';
ctx.fillText('スタートボタンを押してください', canvas.width / 2, canvas.height / 2);
