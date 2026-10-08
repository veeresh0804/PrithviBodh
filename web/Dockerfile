# Build the static bundle, then serve it with nginx.
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json ./
RUN npm install
COPY . ./
# NOTE: do NOT pass VITE_API_URL here — Vite inlines import.meta.env at BUILD
# time, which would bake one environment's API host into every deploy. The
# bundle defaults to same-origin /api/* (proxied to the `api` service by
# nginx.conf), so one image serves all environments. VITE_API_URL is honoured
# only as a local-dev convenience (see src/App.jsx header comment).
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
