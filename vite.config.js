import { defineConfig } from 'vite'
import path from 'path'

export default defineConfig({
    root: 'src', // Point to your existing source directory[1][4]
    build: {
        outDir: '../dist', // Output outside src directory
        emptyOutDir: true,
        rollupOptions: {
            input: {
                main: path.resolve(__dirname, 'src/index.html') // Your entry point
            }
        }
    },
    server: {
        port: 3000
    }
})
