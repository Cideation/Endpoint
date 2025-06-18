FROM node:18-alpine
WORKDIR /app
COPY ./gateway /app
RUN npm install
CMD ["npm", "run", "dev"] 